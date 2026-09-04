"""Choose the detection settings *for this place*, by measuring them.

:mod:`webapp.regions` tabulates what is genuinely regional — currency, tariff,
grid factor, the plausible band a threshold must fall in. This module handles
the part that cannot be tabulated: **where to cut the probability map here**,
and **how hard to close the mask here**.

Three sources of evidence, cheapest first. Each one only runs if the one before
it left room for doubt, and each one records what it did.

1. **Prior** — ``coverage.suggested_threshold`` clamped into the region's band.
   Free, offline, always available. This is what the app did before, and it is
   still the floor everything else is measured against.

2. **Histogram self-calibration** — Otsu's method on the model's own probability
   map. Free (the map already exists), no network, works anywhere on Earth.

   The argument: a well-calibrated segmenter produces a *bimodal* probability
   map — a mass near 0 for background and a mass near 1 for roof — and the right
   cut is the valley between them. Out of distribution the model is
   under-confident, so the roof mode slides down from 0.95 toward 0.55 and the
   valley slides with it. Finding the valley therefore tracks the model's own
   confidence collapse without needing a single label. It is guarded hard: the
   histogram must actually *be* bimodal (Otsu separability), the result must
   imply a plausible amount of roof, and it can only move the prior so far.

3. **Reference calibration** — anchor on real building footprints from
   OpenStreetMap and pick the threshold that recovers them.

   OSM is used **for recall only, never for precision**, and that asymmetry is
   the whole reason it is safe to use. OSM in India is badly *incomplete* — a
   dense Bangalore block may have a fifth of its houses mapped — so "detected
   something OSM does not have" says nothing. But "OSM has a building here" is
   drawn by a human and is almost always true, so "the model missed a building
   OSM has" is real evidence. Calibrating to *recall a known-real set* is
   immune to the incompleteness that would wreck an IoU or precision target.

   This is also the only thing here that can say **thresholding will not fix
   this area**: if recall is still poor at the bottom of the band, the problem
   is the model, not the cut point, and the honest output is to say so rather
   than to keep sliding the threshold down and manufacture false positives out
   of tree canopy.

Reference calibration is the only step that touches the network, so it is an
explicit user action (``POST /api/calibrate``) rather than something an ordinary
analysis waits on. Its result is cached per ~5 km cell, and every later analysis
in that cell picks the cached value up for free.

**Other reference sources.** OSM is the default because it is global, free,
key-less, and human-drawn. For bulk offline work the better sources are Google
Open Buildings (CC-BY-4.0, ~1.8 B footprints across the Global South, including
all of India) and Microsoft's Global ML Building Footprints — but both are
model-generated, so they are fine for *calibration* and must never be used as an
evaluation set. See ``webapp/README.md``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from webapp.config import CACHE_DIR, MIN_BUILDING_AREA_M2, MORPH_KERNEL_PX

__all__ = [
    "Calibration",
    "otsu_threshold",
    "self_calibrate",
    "choose_morph_kernel",
    "fetch_reference_buildings",
    "reference_calibrate",
    "cached_calibration",
    "store_calibration",
    "plan_threshold",
]


# --------------------------------------------------------------------------- #
# Tunables — every one of these is a guard rail, not a fitted parameter
# --------------------------------------------------------------------------- #
# Otsu's separability (eta) below this means the probability map is not bimodal
# enough for a valley to be meaningful, so the histogram step abstains.
#
# 0.88 looks high because eta does not start at 0 for a shapeless histogram: a
# single Gaussian already scores 0.64 and a uniform distribution scores exactly
# 0.75, since Otsu will always find *some* split. Only a genuine two-mode map
# clears 0.9. A lower floor here does not admit weaker evidence — it admits
# noise.
MIN_SEPARABILITY = 0.88
# A cut implying less roof than this or more than this is not believable for an
# AOI someone deliberately drew over buildings.
PLAUSIBLE_POSITIVE_FRAC = (0.01, 0.75)
# The histogram step may never move the prior further than this. It is evidence,
# not an oracle.
MAX_HISTOGRAM_SHIFT = 0.12

# Reference calibration needs enough footprints for a percentile to mean
# anything. Below this it abstains rather than fitting to a handful of roofs.
MIN_REFERENCE_BUILDINGS = 12
# Aim to recover this fraction of known-real buildings. Not 1.0: OSM carries
# demolished buildings, gross misalignments and the occasional mis-tag, and
# chasing the last few percent would drag the threshold to the floor.
TARGET_REFERENCE_RECALL = 0.90
# Below this recall at the *bottom* of the band, the answer is not a threshold.
NEEDS_FINETUNING_RECALL = 0.60
# A footprint smaller than this many mosaic pixels has too few samples for a
# median to be stable.
MIN_REFERENCE_PIXELS = 40

CALIBRATION_TTL_S = 30 * 24 * 3600
CACHE_CELL_DEG = 0.05          # ~5.5 km — one calibration per neighbourhood

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OVERPASS_TIMEOUT_S = 45.0
# Overpass is a shared free service; a runaway bbox is rude and slow. ~4 km^2 is
# far larger than any AOI the tile cap allows.
MAX_REFERENCE_BBOX_DEG = 0.06


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #
@dataclass
class Calibration:
    """The chosen settings plus a full account of how they were chosen."""

    threshold: float
    prior: float
    band: tuple[float, float]
    source: str                      # prior | histogram | reference | user
    confidence: str                  # none | low | medium | high
    morph_kernel_px: int = MORPH_KERNEL_PX
    verdict: str | None = None       # only reference calibration sets this
    note: str = ""
    steps: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band"] = list(self.band)
        d["threshold"] = round(self.threshold, 3)
        d["prior"] = round(self.prior, 3)
        return d


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# 2. Histogram self-calibration
# --------------------------------------------------------------------------- #
def otsu_threshold(probs: np.ndarray, nbins: int = 256) -> tuple[float, float]:
    """Otsu's cut on a probability map. Returns ``(threshold, separability)``.

    ``separability`` is eta = between-class variance / total variance, in
    [0, 1]. It is the part that matters: a clean two-mode map scores well above
    0.8, and a map with no real valley scores low and should be ignored.
    """
    flat = np.asarray(probs, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.5, 0.0

    hist, _ = np.histogram(flat, bins=nbins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.5, 0.0
    p /= total

    centres = (np.arange(nbins) + 0.5) / nbins
    omega = np.cumsum(p)                       # class-0 mass up to bin k
    mu = np.cumsum(p * centres)                # class-0 first moment
    mu_t = float(mu[-1])

    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12,
                           (mu_t * omega - mu) ** 2 / np.maximum(denom, 1e-12),
                           0.0)
    # When the two modes are well separated, every cut inside the empty gap
    # scores identically and sigma_b has a flat plateau. argmax would return the
    # plateau's left edge — pinned to wherever the background tail happens to
    # end — which makes the threshold insensitive to where the roof mode
    # actually sits, and that sensitivity is the entire point. Take the middle
    # of the plateau instead: the centre of the gap.
    best = float(sigma_b.max())
    plateau = np.flatnonzero(sigma_b >= best * (1.0 - 1e-9))
    k = int(plateau[len(plateau) // 2])

    sigma_total = float((p * (centres - mu_t) ** 2).sum())
    eta = float(sigma_b[k] / sigma_total) if sigma_total > 1e-12 else 0.0
    return float(centres[k]), _clamp(eta, 0.0, 1.0)


def self_calibrate(probs: np.ndarray, prior: float,
                   band: tuple[float, float]) -> dict:
    """Nudge ``prior`` toward the valley in the model's own probability map.

    Returns a dict with ``threshold`` (possibly unchanged), ``used``, ``reason``
    and the diagnostics behind the decision. Never moves further than
    ``MAX_HISTOGRAM_SHIFT``, never leaves ``band``, and abstains whenever the
    evidence is weak — the prior is a reasonable answer and this step only earns
    the right to override it by producing a clearly bimodal histogram.
    """
    lo, hi = band
    otsu, eta = otsu_threshold(probs)
    pos_frac = float((probs > otsu).mean()) if probs.size else 0.0

    diag = {
        "otsu": round(otsu, 3),
        "separability": round(eta, 3),
        "positive_fraction_at_otsu": round(pos_frac, 4),
        "prob_p50": round(float(np.median(probs)), 3) if probs.size else 0.0,
        "prob_p99": round(float(np.percentile(probs, 99)), 3) if probs.size else 0.0,
        "mass_above_0.9": round(float((probs > 0.9).mean()), 4) if probs.size else 0.0,
    }

    if eta < MIN_SEPARABILITY:
        return {"threshold": prior, "used": False, **diag,
                "reason": (f"the probability map is not clearly bimodal "
                           f"(separability {eta:.2f} < {MIN_SEPARABILITY}), so "
                           f"the prior was kept")}

    pmin, pmax = PLAUSIBLE_POSITIVE_FRAC
    if not (pmin <= pos_frac <= pmax):
        return {"threshold": prior, "used": False, **diag,
                "reason": (f"the histogram cut implies {pos_frac:.0%} of the "
                           f"area is roof, which is outside the believable "
                           f"{pmin:.0%}-{pmax:.0%} range, so the prior was kept")}

    # Weight by how convincing the valley is, and never hand it full authority:
    # the prior encodes measured out-of-distribution behaviour that a single
    # AOI's histogram does not know about.
    weight = _clamp((eta - MIN_SEPARABILITY) / (1.0 - MIN_SEPARABILITY), 0.0, 1.0) * 0.6
    blended = (1.0 - weight) * prior + weight * otsu
    shifted = _clamp(blended,
                     prior - MAX_HISTOGRAM_SHIFT, prior + MAX_HISTOGRAM_SHIFT)
    final = round(_clamp(shifted, lo, hi), 2)

    moved = abs(final - prior) >= 0.005
    reason = (f"the probability map is bimodal (separability {eta:.2f}) with "
              f"its valley at {otsu:.2f}, so the threshold moved "
              f"{prior:.2f} -> {final:.2f}"
              if moved else
              f"the probability map is bimodal (separability {eta:.2f}) and "
              f"its valley at {otsu:.2f} agrees with the prior, which stays "
              f"at {prior:.2f}")
    return {"threshold": final, "used": moved,
            "weight": round(weight, 3), **diag, "reason": reason}


# --------------------------------------------------------------------------- #
# Morphology: how hard to close the mask here
# --------------------------------------------------------------------------- #
def choose_morph_kernel(mask: np.ndarray, metres_per_px: float,
                        typical_building_m2: float,
                        default: int = MORPH_KERNEL_PX) -> tuple[int, dict]:
    """Pick the open/close kernel from the built form actually on the ground.

    A close with a ``k``-pixel kernel bridges gaps up to roughly ``k *
    metres_per_px`` metres. At the serving resolution of ~0.29 m/px the shipped
    ``k=3`` bridges ~0.9 m — wider than the alley between two Indian row houses,
    which is exactly the known failure where neighbouring houses merge into one
    polygon. Small built form therefore wants a smaller kernel; warehouse roofs
    with rooftop plant want a larger one.

    Uses connected components on the *raw* mask, which is cheap (milliseconds)
    and, unlike full vectorisation, happens before any closing has already
    merged the things being measured.
    """
    diag: dict = {"metres_per_px": round(metres_per_px, 3),
                  "typical_building_m2": typical_building_m2}

    m = np.ascontiguousarray(mask.astype(np.uint8))
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    px_area = max(metres_per_px * metres_per_px, 1e-9)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) * px_area
        areas = areas[areas >= MIN_BUILDING_AREA_M2]
    else:
        areas = np.empty(0)

    # Take the *smaller* of what was measured and what the region expects.
    # The measurement is biased upward by exactly the defect this kernel is
    # meant to reduce: when the model emits two adjacent houses as one blob,
    # the measured median doubles, which would argue for a wider kernel and
    # make the merging worse. Measured on a Bangalore block, 2026-09-04: the
    # median component read 119 m^2 where the regional expectation is 60,
    # because neighbours had already merged. The bias only ever runs one way,
    # so the measurement is only ever allowed to argue downward.
    #
    # The asymmetry is deliberate. Under-closing splits one roof into pieces
    # and barely moves total area, which is what the estimate actually uses;
    # over-closing merges neighbours and is a known, named defect.
    if areas.size >= 5:
        measured = float(np.median(areas))
        scale = min(measured, float(typical_building_m2))
        diag["measured_median_m2"] = round(measured, 1)
        diag["source"] = ("measured" if scale == measured
                          else "regional prior (lower than measured)")
        diag["n_components"] = int(areas.size)
    else:
        scale = float(typical_building_m2)
        diag["source"] = "regional prior"
        diag["n_components"] = int(areas.size)
    diag["median_component_m2"] = round(scale, 1)

    if scale < 80.0:
        kernel = 2
        why = (f"median detected footprint is {scale:.0f} m^2 — small, densely "
               f"packed housing, so the kernel drops to 2 px (~"
               f"{2 * metres_per_px:.1f} m) to stop neighbouring houses closing "
               f"into one polygon")
    elif scale > 400.0:
        kernel = 4
        why = (f"median detected footprint is {scale:.0f} m^2 — large roofs, so "
               f"the kernel rises to 4 px (~{4 * metres_per_px:.1f} m) to bridge "
               f"rooftop plant and skylights")
    else:
        kernel = default
        why = (f"median detected footprint is {scale:.0f} m^2 — the default "
               f"{default} px kernel is right")

    diag["kernel_px"] = kernel
    diag["bridges_m"] = round(kernel * metres_per_px, 2)
    diag["reason"] = why
    return kernel, diag


# --------------------------------------------------------------------------- #
# 3. Reference calibration against real footprints
# --------------------------------------------------------------------------- #
def _overpass_query(west: float, south: float, east: float, north: float) -> str:
    return (f"[out:json][timeout:{int(OVERPASS_TIMEOUT_S)}];"
            f'(way["building"]({south:.6f},{west:.6f},{north:.6f},{east:.6f}););'
            f"out geom;")


async def fetch_reference_buildings(west: float, south: float,
                                    east: float, north: float,
                                    endpoints=OVERPASS_ENDPOINTS
                                    ) -> tuple[list[list[tuple[float, float]]], str]:
    """OSM building ways inside the bbox, as lon/lat rings.

    Returns ``(rings, source_label)``. Raises on total failure so the caller can
    report *why* calibration is unavailable rather than silently degrading —
    "Overpass is down" and "there are no buildings here" are very different
    answers and the user deserves the right one.
    """
    if (east - west) > MAX_REFERENCE_BBOX_DEG or (north - south) > MAX_REFERENCE_BBOX_DEG:
        raise ValueError(
            f"area too large to calibrate against OpenStreetMap "
            f"({east - west:.3f}x{north - south:.3f} deg, limit "
            f"{MAX_REFERENCE_BBOX_DEG}). Draw a smaller box.")

    import httpx

    query = _overpass_query(west, south, east, north)
    last: Exception | None = None
    for url in endpoints:
        try:
            async with httpx.AsyncClient(timeout=OVERPASS_TIMEOUT_S) as client:
                r = await client.post(url, data={"data": query},
                                      headers={"User-Agent":
                                               "rooftop-solar-potential-detection/1.0"})
                r.raise_for_status()
                payload = r.json()
            break
        except Exception as exc:                         # try the next mirror
            last = exc
    else:
        # Overpass signals rate limiting with a bare timeout or a 429 and no
        # body, so str(exc) is routinely empty. Name the type or the user is
        # told "unreachable: " and nothing else.
        detail = f"{type(last).__name__}: {last}".rstrip(": ")
        raise RuntimeError(
            f"OpenStreetMap (Overpass) could not be reached ({detail}). It is "
            f"a free shared service and rate-limits bursts — wait a minute and "
            f"try again.")

    rings: list[list[tuple[float, float]]] = []
    for el in payload.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 4:                                # a closed way needs 4 nodes
            continue
        ring = [(float(p["lon"]), float(p["lat"])) for p in geom
                if "lon" in p and "lat" in p]
        if len(ring) >= 4:
            rings.append(ring)
    return rings, "OpenStreetMap building footprints"


def _ring_scores(rings, probs: np.ndarray, grid) -> list[dict]:
    """Median model probability inside each reference footprint.

    Median, not mean: half the pixels above the threshold is a good proxy for
    "this footprint will survive vectorisation as a polygon", and a median
    shrugs off a footprint whose corner overlaps a road.
    """
    h, w = probs.shape[:2]
    out: list[dict] = []
    for ring in rings:
        pts = [grid.lonlat_to_pixel(lon, lat) for lon, lat in ring]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0 = int(math.floor(min(xs))), int(math.floor(min(ys)))
        x1, y1 = int(math.ceil(max(xs))), int(math.ceil(max(ys)))
        # Only footprints wholly inside the mosaic: a building clipped by the
        # edge has a truncated pixel set and a meaningless median.
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h or x1 <= x0 or y1 <= y0:
            continue

        poly = np.array([[int(round(px)) - x0, int(round(py)) - y0]
                         for px, py in pts], dtype=np.int32)
        stencil = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillPoly(stencil, [poly], 1)
        sel = stencil.astype(bool)
        n_px = int(sel.sum())
        if n_px < MIN_REFERENCE_PIXELS:
            continue
        vals = probs[y0:y1 + 1, x0:x1 + 1][sel]
        out.append({"median": float(np.median(vals)),
                    "mean": float(vals.mean()),
                    "n_px": n_px})
    return out


SIZE_BANDS_M2 = ((0, 50), (50, 100), (100, 200), (200, 500), (500, None))


def _recall_by_size(scores: list[dict], threshold: float,
                    metres_per_px: float) -> list[dict]:
    """Recall split by footprint size.

    Worth its few lines: a flat 30% recall and a 30% that is 8% on houses and
    46% on large roofs are completely different diagnoses, and only the second
    one tells a user which of their buildings the number can be trusted for.
    """
    px_area = metres_per_px * metres_per_px
    areas = np.array([s["n_px"] for s in scores], dtype=np.float64) * px_area
    meds = np.array([s["median"] for s in scores], dtype=np.float64)

    out = []
    for lo, hi in SIZE_BANDS_M2:
        sel = (areas >= lo) if hi is None else ((areas >= lo) & (areas < hi))
        if not sel.any():
            continue
        out.append({
            "from_m2": lo,
            "to_m2": hi,
            "n": int(sel.sum()),
            "recall": round(float((meds[sel] > threshold).mean()), 3),
            "median_probability": round(float(np.median(meds[sel])), 4),
        })
    return out


def reference_calibrate(rings, probs: np.ndarray, grid,
                        band: tuple[float, float],
                        prior: float,
                        target_recall: float = TARGET_REFERENCE_RECALL,
                        metres_per_px: float | None = None) -> dict:
    """Pick the threshold that recovers ``target_recall`` of known-real roofs.

    A footprint counts as recovered when its median probability clears the
    threshold. The threshold is then the ``(1 - target_recall)`` quantile of
    those medians — by construction, that fraction of the reference set survives.

    The sweep is returned too, because the shape of recall-versus-threshold is
    the actual diagnosis. Recall that is already poor at the bottom of the band
    means no cut point will help and the model needs fine-tuning here.
    """
    lo, hi = band
    scores = _ring_scores(rings, probs, grid)
    medians = np.array([s["median"] for s in scores], dtype=np.float64)

    sweep = []
    for t in np.arange(lo, hi + 1e-9, 0.025):
        rec = float((medians > t).mean()) if medians.size else 0.0
        sweep.append({"threshold": round(float(t), 3),
                      "reference_recall": round(rec, 3),
                      "positive_fraction": round(float((probs > t).mean()), 4)})

    diag = {
        "reference_found": len(rings),
        "reference_usable": int(medians.size),
        "sweep": sweep,
        "median_of_medians": (round(float(np.median(medians)), 3)
                              if medians.size else None),
        # How many known-real buildings the model does not merely rank low but
        # actively calls background. These are unreachable by any threshold, so
        # this number is the ceiling on what calibration can ever fix.
        "silent_fraction": (round(float((medians < 0.10).mean()), 3)
                            if medians.size else None),
    }

    if medians.size < MIN_REFERENCE_BUILDINGS:
        return {"threshold": prior, "used": False, "verdict": "insufficient_reference",
                **diag,
                "reason": (f"only {medians.size} usable OpenStreetMap footprint"
                           f"{'' if medians.size == 1 else 's'} inside this area "
                           f"(need {MIN_REFERENCE_BUILDINGS}). Buildings here are "
                           f"probably not mapped yet — try a denser or "
                           f"better-mapped block.")}

    raw = float(np.quantile(medians, 1.0 - target_recall))
    chosen = round(_clamp(raw, lo, hi), 2)
    if metres_per_px:
        diag["recall_by_size"] = _recall_by_size(scores, chosen, metres_per_px)
    recall_at_chosen = float((medians > chosen).mean())
    recall_at_floor = float((medians > lo).mean())
    diag.update({
        "raw_quantile": round(raw, 3),
        "reference_recall": round(recall_at_chosen, 3),
        "reference_recall_at_band_floor": round(recall_at_floor, 3),
        "clamped": abs(raw - chosen) > 1e-6,
    })

    if recall_at_floor < NEEDS_FINETUNING_RECALL:
        return {"threshold": chosen, "used": True, "verdict": "needs_finetuning",
                **diag,
                "reason": (f"even at the most permissive threshold this region "
                           f"allows ({lo:.2f}), the model recovers only "
                           f"{recall_at_floor:.0%} of the {medians.size} mapped "
                           f"buildings here. No threshold fixes that — the model "
                           f"has not learnt this kind of rooftop. Treat detected "
                           f"area as a lower bound.")}

    verdict = "calibrated" if recall_at_chosen >= 0.85 else "partial"
    return {"threshold": chosen, "used": True, "verdict": verdict, **diag,
            "reason": (f"threshold {chosen:.2f} recovers {recall_at_chosen:.0%} "
                       f"of the {medians.size} OpenStreetMap buildings mapped in "
                       f"this area (the prior was {prior:.2f}).")}


# --------------------------------------------------------------------------- #
# Cache — one calibration per ~5 km cell, reused for free by later analyses
# --------------------------------------------------------------------------- #
def _cache_dir() -> Path:
    d = Path(CACHE_DIR) / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cell_key(lat: float, lon: float, model: str, zoom: int) -> str:
    """Grid a point to a ~5 km cell so neighbouring AOIs share one calibration."""
    iy = int(math.floor(lat / CACHE_CELL_DEG))
    ix = int(math.floor(lon / CACHE_CELL_DEG))
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)[:48]
    return f"{safe}_z{zoom}_{iy}_{ix}"


def cached_calibration(lat: float, lon: float, model: str,
                       zoom: int) -> dict | None:
    """A stored calibration for this cell, or None if absent/expired/corrupt."""
    path = _cache_dir() / f"{cell_key(lat, lon, model, zoom)}.json"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if time.time() - float(rec.get("created", 0)) > CALIBRATION_TTL_S:
        return None
    if "threshold" not in rec:
        return None
    return rec


def store_calibration(lat: float, lon: float, model: str, zoom: int,
                      record: dict) -> Path:
    """Persist a calibration for this cell. Best-effort; never raises."""
    path = _cache_dir() / f"{cell_key(lat, lon, model, zoom)}.json"
    payload = {**record, "created": time.time(),
               "lat": round(lat, 4), "lon": round(lon, 4),
               "model": model, "zoom": zoom}
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass
    return path


# --------------------------------------------------------------------------- #
# The pipeline the analysis path actually calls
# --------------------------------------------------------------------------- #
def plan_threshold(prior: float, band: tuple[float, float],
                   cached: dict | None) -> Calibration:
    """Step 1 and, if a cached measurement exists, step 3 — before inference.

    Returns a :class:`Calibration` good enough to run the model with.
    :func:`refine_after_inference` then applies the histogram step, which needs
    the probability map that inference produces.
    """
    lo, hi = band
    base = round(_clamp(prior, lo, hi), 2)
    steps = [f"regional prior {prior:.2f}"
             + (f", clamped into this region's band [{lo:.2f}, {hi:.2f}]"
                if abs(base - prior) > 1e-9 else "")]

    if cached is not None:
        thr = round(_clamp(float(cached["threshold"]), lo, hi), 2)
        verdict = cached.get("verdict")
        steps.append(
            f"a stored calibration for this neighbourhood set {thr:.2f} "
            f"({cached.get('reference_usable', '?')} mapped buildings, recall "
            f"{cached.get('reference_recall', '?')})")
        return Calibration(
            threshold=thr, prior=base, band=(lo, hi), source="reference",
            confidence="high" if verdict == "calibrated" else "medium",
            verdict=verdict,
            note=str(cached.get("reason", "")),
            steps=steps,
            diagnostics={"cached": True,
                         "reference_usable": cached.get("reference_usable"),
                         "reference_recall": cached.get("reference_recall"),
                         "age_days": round(
                             (time.time() - float(cached.get("created", 0)))
                             / 86400.0, 1)},
        )

    return Calibration(threshold=base, prior=base, band=(lo, hi),
                       source="prior", confidence="low", steps=steps,
                       note=("No local measurement for this area yet. Run "
                             "“Calibrate to this area” to check the "
                             "threshold against real building footprints."))


def refine_after_inference(cal: Calibration, probs: np.ndarray) -> Calibration:
    """Step 2. Applied only when no measured calibration already won."""
    if cal.source == "reference":
        cal.diagnostics["histogram"] = {
            "skipped": "a measured calibration for this area takes precedence"}
        return cal

    hist = self_calibrate(probs, cal.threshold, cal.band)
    cal.diagnostics["histogram"] = hist
    if hist["used"]:
        cal.threshold = float(hist["threshold"])
        cal.source = "histogram"
        cal.confidence = "medium"
        cal.steps.append(hist["reason"])
    else:
        cal.steps.append(hist["reason"])
    return cal
