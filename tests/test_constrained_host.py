"""Guardrails for a small host — the free tier that the first public deploy hit.

Render's free tier gives **0.15 of a CPU core** and 512 MB. An analysis that
takes ~11 s on a real core takes ~75 s there, and a test-time-augmentation run
(8 passes per window) takes ~10 minutes — long enough that the first user
assumed the page had hung. Two env-controlled valves close that gap:

* ``RSOLAR_DISABLE_TTA`` — the server ignores ``tta=true`` and ``/api/config``
  says so, so the UI can grey the toggle out instead of silently dropping it.
* ``RSOLAR_JOB_TIMEOUT_S`` — a wall-clock cap, checked between sliding windows
  via the progress callback, so a stuck job fails with a readable message
  within one window of the deadline instead of running forever.

Both are read at import time, so these tests reload ``webapp.config`` and
``webapp.app`` under the chosen environment and drive a fresh TestClient.
"""

import importlib
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient  # noqa: E402

from webapp.config import MODELS_DIR  # noqa: E402

needs_model = pytest.mark.skipif(
    not any(MODELS_DIR.glob("*.onnx")),
    reason="no exported model — run scripts/export_onnx.py")

BHOPAL = {"west": 77.4000, "south": 23.2100, "east": 77.4020, "north": 23.2115}


@pytest.fixture
def app_under_env(monkeypatch):
    """(TestClient, app_module) with webapp reloaded under the given env."""
    created = []

    def _make(**env: str):
        for k in ("RSOLAR_DISABLE_TTA", "RSOLAR_JOB_TIMEOUT_S"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        import webapp.config as cfg
        import webapp.app as app_mod
        importlib.reload(cfg)
        importlib.reload(app_mod)

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
        c = TestClient(app_mod.app)
        c.__enter__()
        created.append(c)
        return c, app_mod

    yield _make
    for c in created:
        c.__exit__(None, None, None)
    # Leave the modules in their default state for the rest of the suite.
    import webapp.config as cfg
    import webapp.app as app_mod
    importlib.reload(cfg)
    importlib.reload(app_mod)


def _run(client, jid, timeout_s=120.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {jid} never finished (last: {j['state']})")


def test_config_advertises_tta_by_default(app_under_env):
    client, _ = app_under_env()
    c = client.get("/api/config").json()
    assert c["tta_available"] is True
    assert c["job_timeout_s"] is None


def test_disable_tta_is_reflected_in_config(app_under_env):
    client, _ = app_under_env(RSOLAR_DISABLE_TTA="1")
    assert client.get("/api/config").json()["tta_available"] is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_disable_tta_accepts_the_usual_truthy_spellings(app_under_env, val):
    client, _ = app_under_env(RSOLAR_DISABLE_TTA=val)
    assert client.get("/api/config").json()["tta_available"] is False


@needs_model
def test_tta_request_is_downgraded_not_obeyed_when_disabled(app_under_env):
    client, _ = app_under_env(RSOLAR_DISABLE_TTA="1")
    r = client.post("/api/analyze", json={**{"bounds": BHOPAL}, "tta": True})
    assert r.status_code == 202
    j = _run(client, r.json()["job_id"])
    assert j["state"] == "done", j.get("error")
    assert j["result"]["detection"]["tta"] is False
    assert any("High accuracy" in w for w in j["result"]["warnings"])


@needs_model
def test_tta_still_works_when_not_disabled(app_under_env):
    client, _ = app_under_env()
    r = client.post("/api/analyze", json={**{"bounds": BHOPAL}, "tta": True})
    j = _run(client, r.json()["job_id"])
    assert j["state"] == "done", j.get("error")
    assert j["result"]["detection"]["tta"] is True


@needs_model
def test_job_timeout_stops_a_slow_analysis_with_a_readable_message(
        app_under_env, monkeypatch):
    """A 0.5 s cap plus a deliberately slow model → the job errors, cleanly."""
    client, app_mod = app_under_env(RSOLAR_JOB_TIMEOUT_S="0.5")

    real_predict = app_mod.predict_mask

    def slow_predict(image, bundle, **kw):
        prog = kw.get("progress")
        # Drive the progress callback past the deadline; it raises TimeoutError
        # from inside, exactly as the real per-window loop would.
        for i in range(1, 6):
            if prog:
                prog(i, 5)
            time.sleep(0.2)
        return real_predict(image, bundle, **kw)

    monkeypatch.setattr(app_mod, "predict_mask", slow_predict)

    r = client.post("/api/analyze", json={"bounds": BHOPAL})
    j = _run(client, r.json()["job_id"], timeout_s=30)
    assert j["state"] == "error"
    assert "was stopped" in j["error"]
    assert "smaller" in j["error"]
    # A clean stop, not a crash: the message is user-facing, no exception class.
    assert "Error" not in j["error"] and "Traceback" not in j["error"]


@needs_model
def test_no_timeout_by_default_lets_a_normal_job_finish(app_under_env):
    client, _ = app_under_env()          # RSOLAR_JOB_TIMEOUT_S unset
    r = client.post("/api/analyze", json={"bounds": BHOPAL})
    j = _run(client, r.json()["job_id"])
    assert j["state"] == "done", j.get("error")
