"""API surface and the serving inference path.

Uses FastAPI's TestClient, so no server needs to be running. Anything that would
hit the network (tiles, PVGIS) is stubbed — a test suite that needs the internet
is a test suite that fails on a train.
"""

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient  # noqa: E402

from webapp.app import app  # noqa: E402
from webapp.config import MODELS_DIR  # noqa: E402

HAS_MODEL = any(MODELS_DIR.glob("*.onnx"))
needs_model = pytest.mark.skipif(
    not HAS_MODEL, reason="no exported model — run scripts/export_onnx.py")


@pytest.fixture
def client():
    # As a context manager, so the ASGI portal (and therefore the event loop
    # running the analysis background task) stays alive across requests. A bare
    # TestClient() tears the loop down after each call and the job never
    # advances past its first await.
    with TestClient(app) as c:
        yield c


def wait_for_job(client, jid: str, timeout_s: float = 120.0) -> dict:
    """Poll a job to a terminal state, yielding time to the event loop."""
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {jid} did not finish within {timeout_s}s "
                         f"(last state: {j['state']})")


# --------------------------------------------------------------------------- #
# Read-only endpoints
# --------------------------------------------------------------------------- #
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_and_static_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_config_exposes_what_the_client_needs(client):
    c = client.get("/api/config").json()
    assert c["serving_zoom"] == 19
    assert c["max_tiles"] > 0
    assert c["tile_provider"]["url"].count("{z}") == 1
    assert c["tile_provider"]["attribution"]


def test_assumptions_documents_every_default(client):
    a = client.get("/api/assumptions").json()
    for key in ("packing_factor", "module_efficiency", "tariff_per_kwh",
                "grid_emission_kg_per_kwh"):
        assert key in a["defaults"]
        assert key in a["sources"], f"{key} has no stated source"
    assert a["chain"]


@needs_model
def test_model_card_reports_the_measured_metric(client):
    m = client.get("/api/model").json()
    assert m["architecture"] and m["encoder"]
    assert m["metrics"]["val"]["iou"] > 0.5
    assert m["limitations"], "a model card without limitations is marketing"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bounds", [
    {"west": 10, "south": 10, "east": 5, "north": 20},      # east <= west
    {"west": 10, "south": 20, "east": 15, "north": 10},     # north <= south
    {"west": -200, "south": 10, "east": 15, "north": 20},   # lon out of range
    {"west": 10, "south": -90, "east": 15, "north": 20},    # lat past Mercator
])
def test_invalid_bounds_are_rejected(client, bounds):
    assert client.post("/api/analyze", json={"bounds": bounds}).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("packing_factor", 1.5), ("packing_factor", 0.0),
    ("module_efficiency", 0.9), ("threshold", 1.5), ("zoom", 25),
    ("grid_emission_kg_per_kwh", -1),
])
def test_out_of_range_parameters_are_rejected(client, field, value):
    body = {"bounds": {"west": 77.40, "south": 23.21, "east": 77.41, "north": 23.22},
            field: value}
    assert client.post("/api/analyze", json=body).status_code == 422


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


@needs_model
def test_oversized_aoi_fails_the_job_with_a_useful_message(client):
    # A whole city at z19 is far over the tile cap.
    r = client.post("/api/analyze", json={
        "bounds": {"west": 77.30, "south": 23.15, "east": 77.55, "north": 23.35}})
    assert r.status_code == 202
    j = wait_for_job(client, r.json()["job_id"])
    assert j["state"] == "error"
    assert "too large" in j["error"]
    assert str(client.get("/api/config").json()["max_tiles"]) in j["error"]


# --------------------------------------------------------------------------- #
# Full pipeline, with the network stubbed
# --------------------------------------------------------------------------- #
@needs_model
def test_analyze_end_to_end_without_network(client, monkeypatch):
    """Synthetic imagery: bright blocks on dark ground, no HTTP anywhere."""
    from webapp import app as app_mod

    async def fake_fetch(grid, url, progress=None):
        img = np.full((grid.height_px, grid.width_px, 3), 40, dtype=np.uint8)
        img[100:400, 100:400] = 220
        if progress:
            progress(grid.n_tiles, grid.n_tiles)
        return img, 0

    monkeypatch.setattr(app_mod, "fetch_mosaic", fake_fetch)
    monkeypatch.setattr(
        "webapp.solar.pvgis_yield",
        lambda lat, lon, p: {"annual_kwh_per_kwp": 1500.0,
                             "monthly_kwh_per_kwp": [125.0] * 12,
                             "optimal_tilt_deg": 20.0, "azimuth_deg": 0.0,
                             "source": "stub", "ok": True})

    r = client.post("/api/analyze", json={
        "bounds": {"west": 77.4000, "south": 23.2100,
                   "east": 77.4020, "north": 23.2115},
        "packing_factor": 0.8})
    assert r.status_code == 202
    j = wait_for_job(client, r.json()["job_id"])
    assert j["state"] == "done", j.get("error")

    res = j["result"]
    assert res["geojson"]["type"] == "FeatureCollection"
    assert res["summary"]["packing_factor"] == 0.8
    assert res["coverage"]["level"] == "untested"      # Bhopal
    assert res["region"]["key"] == "in"
    # The detection settings must be chosen for this place and be auditable.
    cal = res["calibration"]
    assert cal["source"] in ("prior", "histogram", "reference")
    assert cal["band"][0] <= res["detection"]["threshold"] <= cal["band"][1]
    assert cal["steps"], "a calibration with no audit trail is not auditable"
    assert res["detection"]["morph_kernel_px"] >= 1
    assert res["imagery"]["zoom"] == 19
    assert res["model"]["architecture"]
    # usable area must be the packing factor times roof area, always
    s = res["summary"]
    assert s["usable_area_m2"] == pytest.approx(s["roof_area_m2"] * 0.8, rel=1e-2)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_region_profile_reports_local_defaults_and_a_threshold(client):
    r = client.get("/api/region-profile", params={"lat": 12.97, "lon": 77.59})
    assert r.status_code == 200
    body = r.json()
    assert body["region"]["key"] == "in"
    assert body["region"]["economics"]["currency"] == "INR"
    assert body["coverage"]["level"] == "untested"
    lo, hi = body["region"]["threshold_band"]
    assert lo <= body["calibration"]["threshold"] <= hi
    # Every pre-filled number must carry a stated source, same rule as
    # /api/assumptions.
    for key in body["region"]["economics"]:
        if key in ("currency", "currency_symbol"):
            continue
        assert key in body["region"]["sources"], f"{key} has no stated source"


def test_region_profile_puts_an_unlisted_location_in_no_ones_currency(client):
    body = client.get("/api/region-profile",
                      params={"lat": -23.55, "lon": -46.63}).json()   # Sao Paulo
    assert body["region"]["key"] == "global"
    assert body["region"]["economics_confidence"] == "none"
    assert body["region"]["economics"]["currency"] != "INR"


def test_region_profile_rejects_impossible_coordinates(client):
    assert client.get("/api/region-profile",
                      params={"lat": 991, "lon": 0}).status_code == 422


@needs_model
def test_sliding_window_returns_the_input_geometry():
    from webapp.inference import load_model, predict_mask

    bundle = load_model()
    # Deliberately not a multiple of the window, to exercise the padding path.
    img = np.random.randint(0, 255, (600, 730, 3), dtype=np.uint8)
    mask, probs = predict_mask(img, bundle)
    assert mask.shape == (600, 730)
    assert probs.shape == (600, 730)
    assert mask.dtype == bool
    assert 0.0 <= probs.min() and probs.max() <= 1.0


@needs_model
def test_smaller_than_one_window_is_padded_not_rejected():
    from webapp.inference import load_model, predict_mask

    bundle = load_model()
    img = np.random.randint(0, 255, (120, 90, 3), dtype=np.uint8)
    mask, probs = predict_mask(img, bundle)
    assert mask.shape == (120, 90)


@needs_model
def test_threshold_monotonically_shrinks_the_mask():
    from webapp.inference import load_model, predict_mask

    bundle = load_model()
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    low, _ = predict_mask(img, bundle, threshold=0.2)
    high, _ = predict_mask(img, bundle, threshold=0.8)
    assert high.sum() <= low.sum()


@needs_model
def test_sidecar_carries_the_normalisation_constants():
    """F-01 insurance: preprocessing must come from the manifest, not source."""
    from webapp.inference import load_model

    b = load_model()
    assert len(b.manifest["mean"]) == 3
    assert len(b.manifest["std"]) == 3
    assert b.window % 32 == 0
    assert 0 < b.threshold < 1
