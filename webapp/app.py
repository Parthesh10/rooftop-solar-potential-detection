"""FastAPI service: AOI -> roof polygons -> PV potential.

One process serves both the API and the UI, so running this locally is a single
command with no Node toolchain:

    python -m webapp            # http://127.0.0.1:8000

Analysis is a job, not a blocking request. A 0.3 km^2 AOI is ~64 tiles and tens
of seconds of tile fetching plus CPU inference; holding an HTTP connection open
for that is fragile and gives the user no progress. ``POST /api/analyze``
returns a job id immediately and the UI polls ``GET /api/jobs/{id}``.
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from webapp import coverage, geometry, solar
from webapp.config import (
    ASSUMPTION_SOURCES,
    MAX_TILES,
    SERVING_ZOOM,
    WEBAPP_ROOT,
    SolarParams,
    tile_provider_config,
)
from webapp.inference import load_model, predict_mask
from webapp.tiles import fetch_mosaic, metres_per_pixel, tile_grid_for_bounds

app = FastAPI(
    title="Rooftop Solar Potential Detection",
    description="Segment rooftops from aerial imagery and estimate PV potential.",
    version="1.0.0",
)

_MODEL = None
_MODEL_ERROR: str | None = None


def get_model():
    """Lazy singleton: a failed load must not stop the UI from rendering."""
    global _MODEL, _MODEL_ERROR
    if _MODEL is None and _MODEL_ERROR is None:
        try:
            _MODEL = load_model()
        except Exception as exc:
            _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
    if _MODEL is None:
        raise HTTPException(503, f"model unavailable — {_MODEL_ERROR}")
    return _MODEL


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Bounds(BaseModel):
    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-85.05, le=85.05)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-85.05, le=85.05)

    @field_validator("east")
    @classmethod
    def _lon_order(cls, v, info):
        west = info.data.get("west")
        if west is not None and v <= west:
            raise ValueError("east must be greater than west")
        return v

    @field_validator("north")
    @classmethod
    def _lat_order(cls, v, info):
        south = info.data.get("south")
        if south is not None and v <= south:
            raise ValueError("north must be greater than south")
        return v


class AnalyzeRequest(BaseModel):
    bounds: Bounds
    zoom: int = Field(SERVING_ZOOM, ge=16, le=21)
    threshold: float | None = Field(None, ge=0.05, le=0.95)
    packing_factor: float = Field(0.75, ge=0.1, le=1.0)
    module_efficiency: float = Field(0.20, ge=0.05, le=0.30)
    system_losses_pct: float = Field(14.0, ge=0.0, le=50.0)
    tilt_deg: float | None = Field(None, ge=0, le=90)
    azimuth_deg: float = Field(0.0, ge=-180, le=180)
    tariff_per_kwh: float = Field(6.5, ge=0.0)
    cost_per_kwp: float = Field(45000.0, ge=0.0)
    subsidy: float = Field(0.0, ge=0.0)
    grid_emission_kg_per_kwh: float = Field(0.71, ge=0.0, le=2.0)
    currency: str = "INR"
    currency_symbol: str = "₹"
    tile_provider: str | None = None

    def solar_params(self) -> SolarParams:
        return SolarParams(
            packing_factor=self.packing_factor,
            module_efficiency=self.module_efficiency,
            system_losses_pct=self.system_losses_pct,
            tilt_deg=self.tilt_deg,
            azimuth_deg=self.azimuth_deg,
            currency=self.currency,
            currency_symbol=self.currency_symbol,
            tariff_per_kwh=self.tariff_per_kwh,
            cost_per_kwp=self.cost_per_kwp,
            subsidy=self.subsidy,
            grid_emission_kg_per_kwh=self.grid_emission_kg_per_kwh,
        )


@dataclass
class Job:
    id: str
    state: Literal["queued", "fetching", "detecting", "measuring", "done", "error"] = "queued"
    progress: float = 0.0
    message: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {
            "id": self.id, "state": self.state,
            "progress": round(self.progress, 3), "message": self.message,
            "result": self.result, "error": self.error,
            "elapsed_s": round(time.time() - self.created, 1),
        }


JOBS: dict[str, Job] = {}
JOB_TTL_S = 3600


def _reap_jobs() -> None:
    cutoff = time.time() - JOB_TTL_S
    for jid in [j for j, job in JOBS.items() if job.created < cutoff]:
        JOBS.pop(jid, None)


# --------------------------------------------------------------------------- #
# The analysis pipeline
# --------------------------------------------------------------------------- #
async def run_analysis(job: Job, req: AnalyzeRequest) -> None:
    try:
        bundle = get_model()
        provider = tile_provider_config(req.tile_provider)
        zoom = min(req.zoom, provider.get("max_zoom", req.zoom))

        b = req.bounds
        grid = tile_grid_for_bounds(b.west, b.south, b.east, b.north, zoom)
        if grid.n_tiles > MAX_TILES:
            raise ValueError(
                f"area too large: {grid.n_tiles} tiles at z{zoom} "
                f"(limit {MAX_TILES}). Draw a smaller box, or lower the zoom.")

        centre_lat = (b.north + b.south) / 2.0
        centre_lon = (b.east + b.west) / 2.0
        mpp = metres_per_pixel(centre_lat, zoom)

        # --- 1. imagery -----------------------------------------------------
        job.state, job.message = "fetching", f"fetching {grid.n_tiles} tiles"

        def tile_progress(done: int, total: int) -> None:
            job.progress = 0.45 * done / max(total, 1)
            job.message = f"fetching imagery {done}/{total}"

        mosaic, n_failed = await fetch_mosaic(grid, provider["url"], tile_progress)
        if n_failed == grid.n_tiles:
            raise RuntimeError(
                "every tile request failed — no imagery available here at this "
                "zoom, or no internet connection.")

        # --- 2. inference ---------------------------------------------------
        job.state, job.message = "detecting", "running the model"

        def infer_progress(done: int, total: int) -> None:
            job.progress = 0.45 + 0.40 * done / max(total, 1)
            job.message = f"detecting rooftops {done}/{total} windows"

        loop = asyncio.get_running_loop()
        mask, probs = await loop.run_in_executor(
            None,
            lambda: predict_mask(mosaic, bundle, threshold=req.threshold,
                                 progress=infer_progress),
        )

        # --- 3. vectorise + measure ----------------------------------------
        job.state, job.message = "measuring", "measuring roof areas"
        job.progress = 0.88

        buildings = await loop.run_in_executor(
            None, lambda: geometry.mask_to_buildings(mask, probs, grid))

        # Only count roofs whose centroid is inside the drawn AOI: the tile grid
        # is always larger than the box the user drew, and billing them for a
        # neighbour's warehouse would be wrong.
        def inside(bl: geometry.Building) -> bool:
            lons = [p[0] for p in bl.exterior]
            lats = [p[1] for p in bl.exterior]
            cx, cy = sum(lons) / len(lons), sum(lats) / len(lats)
            return b.west <= cx <= b.east and b.south <= cy <= b.north

        buildings = [x for x in buildings if inside(x)]

        params = req.solar_params()
        total_roof = sum(x.area_m2 for x in buildings)
        est = solar.estimate(total_roof, centre_lat, centre_lon, params)
        job.progress = 0.97

        aoi_area = geometry.bounds_area_m2(b.west, b.south, b.east, b.north)
        warnings: list[str] = []
        if n_failed:
            warnings.append(
                f"{n_failed} of {grid.n_tiles} imagery tiles failed to load; "
                f"those areas were treated as blank and may hide roofs.")
        warnings.extend(solar.sanity_check_capacity(est["capacity_kwp"]))
        if not buildings:
            warnings.append(
                "No rooftops detected. The imagery may be cloudy, too low "
                "resolution at this location, or genuinely empty.")

        job.result = {
            "coverage": coverage.coverage_note(centre_lat, centre_lon),
            "geojson": geometry.buildings_to_geojson(buildings,
                                                     params.packing_factor),
            "summary": {
                **est,
                "building_count": len(buildings),
                "aoi_area_m2": round(aoi_area, 0),
                "roof_coverage_pct": round(100 * total_roof / aoi_area, 1)
                if aoi_area > 0 else 0.0,
                "mean_confidence": round(
                    sum(x.confidence for x in buildings) / len(buildings), 3)
                if buildings else 0.0,
            },
            "imagery": {
                "provider": provider["label"],
                "attribution": provider["attribution"],
                "zoom": zoom,
                "tiles": grid.n_tiles,
                "tiles_failed": n_failed,
                "metres_per_pixel": round(mpp, 4),
                "mosaic_px": [grid.width_px, grid.height_px],
            },
            "model": bundle.card(),
            "assumptions": params.to_dict(),
            "warnings": warnings,
            "bounds": b.model_dump(),
        }
        job.state, job.message, job.progress = "done", "complete", 1.0

    except Exception as exc:
        job.state = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "failed"
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    _reap_jobs()
    job = Job(id=uuid.uuid4().hex[:12])
    JOBS[job.id] = job
    asyncio.create_task(run_analysis(job, req))
    return JSONResponse({"job_id": job.id}, status_code=202)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown or expired job")
    return job.public()


@app.get("/api/model")
async def model_card():
    try:
        return get_model().card()
    except HTTPException as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.get("/api/assumptions")
async def assumptions():
    defaults = SolarParams()
    return {
        "defaults": defaults.to_dict(),
        "sources": ASSUMPTION_SOURCES,
        "chain": [
            "detected roof footprint (m²)",
            "× packing factor → usable PV area (m²)",
            "× module efficiency → capacity (kWp)",
            "× PVGIS site yield → annual energy (kWh)",
            "× tariff → annual savings; × grid factor → CO₂ avoided",
        ],
    }


@app.get("/api/config")
async def client_config():
    try:
        provider = tile_provider_config()
        prov = {"name": provider["name"], "label": provider["label"],
                "url": provider["url"], "attribution": provider["attribution"],
                "max_zoom": provider["max_zoom"]}
        err = None
    except ValueError as exc:
        prov, err = None, str(exc)
    try:
        model_ok, model_err = bool(get_model()), None
    except HTTPException as exc:
        model_ok, model_err = False, exc.detail
    return {"serving_zoom": SERVING_ZOOM, "max_tiles": MAX_TILES,
            "tile_provider": prov, "tile_provider_error": err,
            "model_ready": model_ok, "model_error": model_err}


@app.get("/api/health")
async def health():
    return {"status": "ok", "jobs": len(JOBS)}


STATIC_DIR = WEBAPP_ROOT / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
