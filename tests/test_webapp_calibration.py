"""The per-area detection layer: the region table and the calibrators.

Nothing here touches the network. The reference calibrator is exercised against
synthetic footprints placed on a real ``TileGrid``, which is the part worth
testing — the Overpass round trip is I/O, the geometry and the statistics are
where a bug would silently change everyone's estimate.
"""

import time

import numpy as np
import pytest

from webapp import calibration, regions
from webapp.tiles import tile_grid_for_bounds

# A small AOI in Bangalore: 2x2 tiles at z19, so a 512x512 mosaic.
BBOX = (77.5900, 12.9900, 77.5930, 12.9925)


@pytest.fixture
def grid():
    return tile_grid_for_bounds(*BBOX, 19)


# --------------------------------------------------------------------------- #
# The region table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lat,lon,key", [
    (12.97, 77.59, "in"),          # Bangalore
    (23.26, 77.41, "in"),          # Bhopal
    (30.27, -97.74, "us"),         # Austin, a training city
    (48.21, 16.37, "at"),          # Vienna, a training city
    (52.52, 13.40, "de"),          # Berlin
    (-33.87, 151.21, "au"),        # Sydney
    (-1.29, 36.82, "af"),          # Nairobi
    (-23.55, -46.63, "global"),    # Sao Paulo — deliberately unlisted
])
def test_resolves_to_the_narrowest_matching_region(lat, lon, key):
    assert regions.resolve(lat, lon).key == key


def test_sri_lanka_is_carved_out_of_indias_bounding_box():
    # India's rectangle contains Sri Lanka; the narrower entry must win, or a
    # Colombo user is quoted rupees at a Madhya Pradesh tariff.
    lk = regions.resolve(6.93, 79.86)
    assert lk.key == "lk"
    assert lk.currency != "INR"
    assert lk.economics_confidence == "none"
    # It still inherits South Asian built form, which is what detection uses.
    assert lk.typical_building_m2 == regions.resolve(12.97, 77.59).typical_building_m2


def test_narrower_regions_inherit_from_broader_ones():
    at = regions.resolve(48.21, 16.37)
    assert at.matched == ["global", "eu", "at"]
    assert at.currency == "EUR"                  # inherited from eu
    assert at.grid_emission_kg_per_kwh == 0.15   # overridden by at


@pytest.mark.parametrize("region", regions.REGIONS, ids=lambda r: r.key)
def test_every_region_declares_a_sane_band(region):
    profile_values = {**regions.GLOBAL.values, **region.values}
    lo, hi = profile_values["threshold_band"]
    assert 0.05 < lo < hi < 0.95
    assert profile_values["typical_building_m2"] > 0
    assert 0.1 <= profile_values["packing_factor"] <= 1.0
    assert profile_values["economics_confidence"] in ("none", "low", "medium")


def test_profile_serialises_for_the_api():
    d = regions.resolve(12.97, 77.59).to_dict()
    assert set(d) >= {"key", "name", "matched", "threshold_band",
                      "economics", "economics_confidence", "sources"}
    assert set(d["economics"]) == {"currency", "currency_symbol",
                                   "tariff_per_kwh", "cost_per_kwp",
                                   "grid_emission_kg_per_kwh", "packing_factor"}


# --------------------------------------------------------------------------- #
# Otsu / histogram self-calibration
# --------------------------------------------------------------------------- #
def _bimodal(roof_mode: float, roof_frac: float = 0.2, n: int = 200_000,
             spread: float = 0.05, seed: int = 0) -> np.ndarray:
    """A probability map with a background mode near 0 and a roof mode given."""
    rng = np.random.default_rng(seed)
    n_roof = int(n * roof_frac)
    bg = np.clip(rng.normal(0.05, spread, n - n_roof), 0.0, 1.0)
    roof = np.clip(rng.normal(roof_mode, spread, n_roof), 0.0, 1.0)
    return np.concatenate([bg, roof]).astype(np.float32)


def test_otsu_finds_the_valley_of_a_confident_map():
    thr, eta = calibration.otsu_threshold(_bimodal(0.95))
    assert 0.3 < thr < 0.8
    assert eta > 0.85


def test_otsu_valley_slides_down_when_the_model_is_under_confident():
    """The whole premise of histogram self-calibration, stated as a test."""
    confident, _ = calibration.otsu_threshold(_bimodal(0.95))
    unsure, _ = calibration.otsu_threshold(_bimodal(0.55))
    assert unsure < confident


def test_otsu_reports_low_separability_on_a_map_with_no_valley():
    rng = np.random.default_rng(1)
    _, eta = calibration.otsu_threshold(rng.random(100_000).astype(np.float32))
    assert eta < calibration.MIN_SEPARABILITY


def test_self_calibrate_abstains_when_the_map_is_not_bimodal():
    rng = np.random.default_rng(2)
    out = calibration.self_calibrate(rng.random(100_000).astype(np.float32),
                                     prior=0.40, band=(0.28, 0.60))
    assert out["used"] is False
    assert out["threshold"] == 0.40
    assert "bimodal" in out["reason"]


def test_self_calibrate_abstains_when_the_cut_implies_absurd_coverage():
    # Almost every pixel is roof-confident: a valley exists, but acting on it
    # would claim the whole AOI is rooftop.
    probs = _bimodal(0.95, roof_frac=0.95)
    out = calibration.self_calibrate(probs, prior=0.40, band=(0.28, 0.60))
    assert out["used"] is False
    assert "believable" in out["reason"]


def test_self_calibrate_moves_down_for_an_under_confident_map():
    out = calibration.self_calibrate(_bimodal(0.50), prior=0.50,
                                     band=(0.28, 0.60))
    assert out["used"] is True
    assert out["threshold"] < 0.50


def test_self_calibrate_never_moves_further_than_the_cap_or_out_of_band():
    # A map whose valley sits far below the prior: the shift must be capped.
    out = calibration.self_calibrate(_bimodal(0.35, spread=0.02), prior=0.60,
                                     band=(0.28, 0.60))
    assert out["threshold"] >= 0.60 - calibration.MAX_HISTOGRAM_SHIFT - 1e-9
    assert 0.28 <= out["threshold"] <= 0.60


def test_refine_after_inference_defers_to_a_measured_calibration():
    cal = calibration.Calibration(threshold=0.33, prior=0.40, band=(0.28, 0.60),
                                  source="reference", confidence="high")
    out = calibration.refine_after_inference(cal, _bimodal(0.50))
    assert out.threshold == 0.33
    assert out.source == "reference"
    assert "skipped" in out.diagnostics["histogram"]


# --------------------------------------------------------------------------- #
# Morphology
# --------------------------------------------------------------------------- #
def _blobs(size_px: int, count: int = 16, canvas: int = 512) -> np.ndarray:
    """A mask of ``count`` square blobs, well separated."""
    mask = np.zeros((canvas, canvas), dtype=bool)
    step = canvas // int(np.ceil(np.sqrt(count)))
    placed = 0
    for y in range(4, canvas - size_px, step):
        for x in range(4, canvas - size_px, step):
            if placed >= count:
                break
            mask[y:y + size_px, x:x + size_px] = True
            placed += 1
    return mask


def test_small_dense_housing_gets_a_smaller_closing_kernel():
    # 25 px at 0.29 m/px is ~52 m^2 — an Indian row house.
    kernel, diag = calibration.choose_morph_kernel(_blobs(25), 0.29, 60.0)
    assert kernel == 2
    assert diag["source"] == "measured"
    assert diag["bridges_m"] < 0.7      # narrower than a 1 m alley


def test_large_roofs_get_a_larger_closing_kernel():
    # 80 px at 0.29 m/px is ~538 m^2 — a warehouse, in a region that expects
    # large roofs. Both signals have to agree before the kernel widens.
    kernel, _ = calibration.choose_morph_kernel(_blobs(80, count=9), 0.29, 600.0)
    assert kernel == 4


def test_a_measured_median_may_only_argue_for_a_narrower_kernel():
    """Merging inflates the measurement, so it must not widen the kernel.

    Measured on a Bangalore block: the median component read 119 m^2 against a
    regional expectation of 60, purely because neighbouring houses had already
    merged. Trusting the measurement there widens the kernel and merges more.
    """
    kernel, diag = calibration.choose_morph_kernel(_blobs(80, count=9), 0.29, 60.0)
    assert kernel == 2
    assert diag["measured_median_m2"] > diag["median_component_m2"]
    assert "regional prior" in diag["source"]


def test_morphology_falls_back_to_the_regional_prior_when_nothing_is_detected():
    kernel, diag = calibration.choose_morph_kernel(
        np.zeros((256, 256), dtype=bool), 0.29, 60.0)
    assert diag["source"] == "regional prior"
    assert kernel == 2                   # the Indian prior still implies dense


def test_geometry_honours_the_chosen_kernel(grid):
    """Two blobs one pixel apart merge under a wide kernel and not a narrow one."""
    from webapp import geometry

    mask = np.zeros((512, 512), dtype=bool)
    mask[100:160, 100:160] = True
    mask[100:160, 163:223] = True        # a 3 px alley between them
    probs = mask.astype(np.float32)

    wide = geometry.mask_to_buildings(mask, probs, grid, morph_kernel_px=6)
    narrow = geometry.mask_to_buildings(mask, probs, grid, morph_kernel_px=1)
    assert len(narrow) == 2
    assert len(wide) < len(narrow)


# --------------------------------------------------------------------------- #
# Reference calibration
# --------------------------------------------------------------------------- #
def _synthetic_reference(grid, levels, size_px: int = 22):
    """Place one square 'building' per level; return (rings, probs).

    Rings are produced by round-tripping pixel corners through the grid, so the
    test exercises the same lon/lat <-> pixel path the Overpass data takes.
    """
    h, w = grid.height_px, grid.width_px
    probs = np.full((h, w), 0.03, dtype=np.float32)
    rings = []
    per_row = max(int(np.floor((w - 20) / (size_px + 12))), 1)
    for i, level in enumerate(levels):
        x = 10 + (i % per_row) * (size_px + 12)
        y = 10 + (i // per_row) * (size_px + 12)
        probs[y:y + size_px, x:x + size_px] = level
        corners = [(x, y), (x + size_px, y),
                   (x + size_px, y + size_px), (x, y + size_px), (x, y)]
        rings.append([grid.pixel_to_lonlat(px, py) for px, py in corners])
    return rings, probs


def test_reference_calibration_hits_its_recall_target(grid):
    levels = np.linspace(0.20, 0.90, 20)
    rings, probs = _synthetic_reference(grid, levels)

    out = calibration.reference_calibrate(rings, probs, grid,
                                          band=(0.28, 0.60), prior=0.40,
                                          target_recall=0.90)
    assert out["used"] is True
    assert out["verdict"] in ("calibrated", "partial")
    assert out["reference_usable"] == len(levels)
    # By construction the chosen cut keeps ~90% of the known-real footprints.
    assert out["reference_recall"] >= 0.80
    assert 0.28 <= out["threshold"] <= 0.60


def test_reference_calibration_abstains_without_enough_footprints(grid):
    rings, probs = _synthetic_reference(grid, np.linspace(0.4, 0.9, 5))
    out = calibration.reference_calibrate(rings, probs, grid,
                                          band=(0.28, 0.60), prior=0.42)
    assert out["used"] is False
    assert out["verdict"] == "insufficient_reference"
    assert out["threshold"] == 0.42          # the prior is left alone


def test_reference_calibration_says_when_no_threshold_can_help(grid):
    """The Indian-campus case: real buildings the model simply does not see."""
    rings, probs = _synthetic_reference(grid, np.full(20, 0.12))
    out = calibration.reference_calibrate(rings, probs, grid,
                                          band=(0.28, 0.60), prior=0.40)
    assert out["verdict"] == "needs_finetuning"
    assert out["reference_recall_at_band_floor"] < calibration.NEEDS_FINETUNING_RECALL
    assert "No threshold fixes that" in out["reason"]


def test_reference_calibration_ignores_footprints_outside_the_mosaic(grid):
    rings, probs = _synthetic_reference(grid, np.linspace(0.4, 0.9, 14))
    far = [(lon + 1.0, lat + 1.0) for lon, lat in rings[0]]      # ~110 km away
    out = calibration.reference_calibrate(rings + [far], probs, grid,
                                          band=(0.28, 0.60), prior=0.40)
    assert out["reference_found"] == 15
    assert out["reference_usable"] == 14


def test_reference_calibration_reports_recall_by_footprint_size(grid):
    """A flat 30% and a 30% that is 8% on houses are different diagnoses."""
    rings, probs = _synthetic_reference(grid, np.linspace(0.20, 0.90, 20))
    out = calibration.reference_calibrate(rings, probs, grid,
                                          band=(0.28, 0.60), prior=0.40,
                                          metres_per_px=0.291)
    bands = out["recall_by_size"]
    assert bands and sum(b["n"] for b in bands) == out["reference_usable"]
    assert all(0.0 <= b["recall"] <= 1.0 for b in bands)


def test_reference_calibration_counts_buildings_the_model_is_silent_on(grid):
    # Half score ~0.02 (unreachable by any threshold), half score ~0.8.
    levels = np.concatenate([np.full(10, 0.02), np.full(10, 0.80)])
    out = calibration.reference_calibrate(rings=_synthetic_reference(grid, levels)[0],
                                          probs=_synthetic_reference(grid, levels)[1],
                                          grid=grid, band=(0.28, 0.60), prior=0.40)
    assert out["silent_fraction"] == pytest.approx(0.5, abs=0.05)


def test_reference_sweep_is_monotonically_non_increasing(grid):
    rings, probs = _synthetic_reference(grid, np.linspace(0.20, 0.90, 20))
    out = calibration.reference_calibrate(rings, probs, grid,
                                          band=(0.28, 0.60), prior=0.40)
    recalls = [p["reference_recall"] for p in out["sweep"]]
    assert recalls == sorted(recalls, reverse=True)


def test_reference_fetch_refuses_an_absurd_bbox():
    import asyncio

    with pytest.raises(ValueError, match="too large"):
        asyncio.run(calibration.fetch_reference_buildings(77.0, 12.0, 78.0, 13.0))


# --------------------------------------------------------------------------- #
# Cache and planning
# --------------------------------------------------------------------------- #
def test_calibration_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CACHE_DIR", tmp_path)
    calibration.store_calibration(12.97, 77.59, "unetpp_effb0", 19,
                                  {"threshold": 0.34, "verdict": "calibrated",
                                   "reference_recall": 0.91})
    got = calibration.cached_calibration(12.97, 77.59, "unetpp_effb0", 19)
    assert got["threshold"] == 0.34
    # A nearby AOI shares the cell; a distant one does not.
    assert calibration.cached_calibration(12.98, 77.59, "unetpp_effb0", 19)
    assert calibration.cached_calibration(13.60, 77.59, "unetpp_effb0", 19) is None
    # A different model must never inherit another model's calibration.
    assert calibration.cached_calibration(12.97, 77.59, "other", 19) is None


def test_expired_calibrations_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CACHE_DIR", tmp_path)
    path = calibration.store_calibration(12.97, 77.59, "m", 19,
                                         {"threshold": 0.34})
    import json

    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["created"] = time.time() - calibration.CALIBRATION_TTL_S - 1
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert calibration.cached_calibration(12.97, 77.59, "m", 19) is None


def test_a_corrupt_cache_entry_is_ignored_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CACHE_DIR", tmp_path)
    key = calibration.cell_key(12.97, 77.59, "m", 19)
    (tmp_path / "calibration").mkdir(parents=True, exist_ok=True)
    (tmp_path / "calibration" / f"{key}.json").write_text("{not json",
                                                          encoding="utf-8")
    assert calibration.cached_calibration(12.97, 77.59, "m", 19) is None


def test_plan_threshold_clamps_the_prior_into_the_regional_band():
    cal = calibration.plan_threshold(0.50, band=(0.28, 0.45), cached=None)
    assert cal.threshold == 0.45
    assert cal.source == "prior"
    assert "clamped" in cal.steps[0]


def test_plan_threshold_prefers_a_stored_measurement():
    cal = calibration.plan_threshold(
        0.40, band=(0.28, 0.60),
        cached={"threshold": 0.33, "verdict": "calibrated", "created": time.time(),
                "reference_usable": 41, "reference_recall": 0.92})
    assert cal.threshold == 0.33
    assert cal.source == "reference"
    assert cal.confidence == "high"
