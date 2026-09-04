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

from webapp import calibration, coverage, geometry, regions, solar
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
    tta: bool = Field(False, description="8x dihedral test-time augmentation: "
                                         "+1 IoU / +2 precision for 8x the time")
    use_calibration: bool = Field(
        True, description="apply the per-area detection calibration "
                          "(regional band, stored measurement, histogram). "
                          "Ignored when 'threshold' is set by hand.")
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


class CalibrateRequest(BaseModel):
    """Measure the right detection threshold for one AOI against real roofs."""

    bounds: Bounds
    zoom: int = Field(SERVING_ZOOM, ge=16, le=21)
    tta: bool = False
    tile_provider: str | None = None
    target_recall: float = Field(
        0.90, ge=0.50, le=0.99,
        description="fraction of known-real (OpenStreetMap) buildings the "
                    "chosen threshold should recover")


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

        # Classify the AOI first: the decision threshold depends on whether the
        # model has seen anything like this place, and on anything that has
        # actually been measured here. See webapp/calibration.py.
        cov = coverage.coverage_note(centre_lat, centre_lon)
        region = regions.resolve(centre_lat, centre_lon)
        auto_threshold = req.threshold is None
        if auto_threshold:
            cached = (calibration.cached_calibration(
                          centre_lat, centre_lon, bundle.name, zoom)
                      if req.use_calibration else None)
            cal = calibration.plan_threshold(
                coverage.suggested_threshold(cov["level"], bundle.threshold),
                region.threshold_band, cached)
        else:
            cal = calibration.Calibration(
                threshold=req.threshold, prior=req.threshold,
                band=region.threshold_band, source="user", confidence="none",
                note="Threshold set by hand — calibration was not applied.",
                steps=[f"threshold {req.threshold:.2f} set by hand"])
        threshold = cal.threshold

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
        suffix = " (high accuracy, 8 passes)" if req.tta else ""

        def infer_progress(done: int, total: int) -> None:
            job.progress = 0.45 + 0.40 * done / max(total, 1)
            job.message = f"detecting rooftops {done}/{total} windows{suffix}"

        loop = asyncio.get_running_loop()
        mask, probs = await loop.run_in_executor(
            None,
            lambda: predict_mask(mosaic, bundle, threshold=threshold,
                                 tta=req.tta, progress=infer_progress),
        )

        # The probability map is the evidence for the histogram step, so this
        # can only happen now. Re-cutting the same map is free — no second
        # inference pass.
        if auto_threshold and req.use_calibration:
            cal = calibration.refine_after_inference(cal, probs)
            if abs(cal.threshold - threshold) > 1e-9:
                threshold = cal.threshold
                mask = probs > threshold

        # --- 3. vectorise + measure ----------------------------------------
        job.state, job.message = "measuring", "measuring roof areas"
        job.progress = 0.88

        # How hard to close the mask is also local: a 3 px kernel bridges ~0.9 m
        # and merges Indian row houses into one polygon.
        morph_kernel, morph_diag = calibration.choose_morph_kernel(
            mask, mpp, region.typical_building_m2)
        cal.morph_kernel_px = morph_kernel
        cal.diagnostics["morphology"] = morph_diag

        buildings = await loop.run_in_executor(
            None, lambda: geometry.mask_to_buildings(
                mask, probs, grid, morph_kernel_px=morph_kernel))

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
        job.message = "fetching solar resource data"
        est = solar.estimate(total_roof, centre_lat, centre_lon, params)

        # The location's solar exposure, independent of any system design. Runs
        # off the same cache, so it is free on a repeat query in the same area.
        resource = await loop.run_in_executor(
            None, lambda: solar.pvgis_radiation(centre_lat, centre_lon))
        job.progress = 0.97

        aoi_area = geometry.bounds_area_m2(b.west, b.south, b.east, b.north)
        warnings: list[str] = []
        if n_failed:
            warnings.append(
                f"{n_failed} of {grid.n_tiles} imagery tiles failed to load; "
                f"those areas were treated as blank and may hide roofs.")
        warnings.extend(solar.sanity_check_capacity(est["capacity_kwp"]))
        if auto_threshold:
            if cal.verdict == "needs_finetuning":
                warnings.append(cal.note)
            elif cal.source == "reference":
                warnings.append(
                    f"Detection sensitivity was measured for this "
                    f"neighbourhood rather than assumed: {cal.note}")
            else:
                note = coverage.threshold_note(cov["level"], threshold)
                if note:
                    warnings.append(note)
        if not buildings:
            warnings.append(
                "No rooftops detected. The imagery may be cloudy, too low "
                "resolution at this location, or genuinely empty.")

        job.result = {
            "coverage": cov,
            "region": region.to_dict(),
            "calibration": cal.to_dict(),
            "solar_resource": {**resource,
                               "lat": round(centre_lat, 4),
                               "lon": round(centre_lon, 4)},
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
            "detection": {
                "threshold": round(threshold, 2),
                "threshold_auto": auto_threshold,
                "threshold_source": cal.source,
                "model_default_threshold": bundle.threshold,
                "morph_kernel_px": morph_kernel,
                "tta": req.tta,
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


async def run_calibration(job: Job, req: CalibrateRequest) -> None:
    """Measure a threshold for this AOI against OpenStreetMap footprints.

    Separate from ``run_analysis`` because it is the only path that touches a
    third-party API and can take tens of seconds on Overpass — an ordinary
    estimate must not wait on that. The result is cached per ~5 km cell, so it
    is paid for once and every later analysis nearby reads it for free.
    """
    try:
        bundle = get_model()
        provider = tile_provider_config(req.tile_provider)
        zoom = min(req.zoom, provider.get("max_zoom", req.zoom))

        b = req.bounds
        grid = tile_grid_for_bounds(b.west, b.south, b.east, b.north, zoom)
        if grid.n_tiles > MAX_TILES:
            raise ValueError(
                f"area too large to calibrate: {grid.n_tiles} tiles at z{zoom} "
                f"(limit {MAX_TILES}). Draw a smaller box.")

        centre_lat = (b.north + b.south) / 2.0
        centre_lon = (b.east + b.west) / 2.0
        cov = coverage.coverage_note(centre_lat, centre_lon)
        region = regions.resolve(centre_lat, centre_lon)
        prior = coverage.suggested_threshold(cov["level"], bundle.threshold)

        # Reference footprints first: with none, there is nothing to calibrate
        # against and an expensive inference pass would be wasted.
        job.state, job.message = "fetching", "asking OpenStreetMap what is here"
        job.progress = 0.05
        rings, ref_source = await calibration.fetch_reference_buildings(
            b.west, b.south, b.east, b.north)
        job.progress = 0.15
        job.message = f"{len(rings)} mapped buildings found"

        def tile_progress(done: int, total: int) -> None:
            job.progress = 0.15 + 0.35 * done / max(total, 1)
            job.message = f"fetching imagery {done}/{total}"

        mosaic, n_failed = await fetch_mosaic(grid, provider["url"], tile_progress)
        if n_failed == grid.n_tiles:
            raise RuntimeError("every tile request failed — no imagery here.")

        job.state, job.message = "detecting", "running the model"

        def infer_progress(done: int, total: int) -> None:
            job.progress = 0.50 + 0.40 * done / max(total, 1)
            job.message = f"scoring rooftops {done}/{total} windows"

        loop = asyncio.get_running_loop()
        # The threshold passed here is irrelevant: only ``probs`` is used, and
        # deciding where to cut it is the entire point of this job.
        _, probs = await loop.run_in_executor(
            None,
            lambda: predict_mask(mosaic, bundle, threshold=0.5, tta=req.tta,
                                 progress=infer_progress),
        )

        job.state, job.message = "measuring", "matching detections to mapped roofs"
        job.progress = 0.93
        result = await loop.run_in_executor(
            None,
            lambda: calibration.reference_calibrate(
                rings, probs, grid, region.threshold_band, prior,
                target_recall=req.target_recall,
                metres_per_px=metres_per_pixel(centre_lat, zoom)),
        )

        cal = calibration.Calibration(
            threshold=float(result["threshold"]),
            prior=prior,
            band=region.threshold_band,
            source="reference" if result["used"] else "prior",
            confidence=("high" if result.get("verdict") == "calibrated"
                        else "medium" if result["used"] else "low"),
            verdict=result.get("verdict"),
            note=result["reason"],
            steps=[f"regional prior {prior:.2f}", result["reason"]],
            diagnostics={k: v for k, v in result.items()
                         if k not in ("threshold", "used", "reason", "verdict")},
        )

        stored = None
        if result["used"]:
            stored = str(calibration.store_calibration(
                centre_lat, centre_lon, bundle.name, zoom,
                {"threshold": cal.threshold,
                 "verdict": cal.verdict,
                 "reason": cal.note,
                 "reference_usable": result.get("reference_usable"),
                 "reference_recall": result.get("reference_recall"),
                 "reference_source": ref_source,
                 "target_recall": req.target_recall}))

        job.result = {
            "coverage": cov,
            "region": region.to_dict(),
            "calibration": cal.to_dict(),
            "reference": {
                "source": ref_source,
                "found": result.get("reference_found"),
                "usable": result.get("reference_usable"),
                "target_recall": req.target_recall,
                "caveat": ("OpenStreetMap is used for recall only. It is "
                           "incomplete in many places, so a detection it does "
                           "not have is not evidence of a false positive — but "
                           "a building it does have was drawn by a human, so a "
                           "miss is real."),
            },
            "cached_to": stored,
            "imagery": {"provider": provider["label"], "zoom": zoom,
                        "tiles": grid.n_tiles, "tiles_failed": n_failed},
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


@app.post("/api/calibrate")
async def calibrate(req: CalibrateRequest):
    """Measure the detection threshold for this AOI. Returns a job id."""
    _reap_jobs()
    job = Job(id=uuid.uuid4().hex[:12])
    JOBS[job.id] = job
    asyncio.create_task(run_calibration(job, req))
    return JSONResponse({"job_id": job.id}, status_code=202)


@app.get("/api/region-profile")
async def region_profile(lat: float, lon: float):
    """What is known about this place before any pixel is looked at.

    Regional money/grid defaults, the threshold band, the coverage note, and any
    calibration already measured nearby. The UI calls this whenever the map
    settles, so the Assumptions panel can start in the right currency rather
    than showing a Brazilian user rupees.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(422, "lat/lon out of range")

    region = regions.resolve(lat, lon)
    cov = coverage.coverage_note(lat, lon)
    try:
        bundle = get_model()
        prior = coverage.suggested_threshold(cov["level"], bundle.threshold)
        cached = calibration.cached_calibration(lat, lon, bundle.name, SERVING_ZOOM)
    except HTTPException:
        prior, cached = coverage.suggested_threshold(cov["level"]), None

    cal = calibration.plan_threshold(prior, region.threshold_band, cached)
    return {"region": region.to_dict(), "coverage": cov,
            "calibration": cal.to_dict(),
            "lat": round(lat, 4), "lon": round(lon, 4)}


@app.get("/api/model")
async def model_card():
    try:
        return get_model().card()
    except HTTPException as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.get("/api/solar-resource")
async def solar_resource(lat: float, lon: float):
    """Monthly solar exposure at a point — usable without running a detection.

    Split out so the UI can show the resource as soon as the map settles, before
    anyone draws a box.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(422, "lat/lon out of range")
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lambda: solar.pvgis_radiation(lat, lon))
    return {**data, "lat": round(lat, 4), "lon": round(lon, 4)}


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
