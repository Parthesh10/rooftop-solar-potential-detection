"""Tests for the /api/analyze rate limit.

webapp/README.md §Deploying names this as a prerequisite for going public. The
point is arithmetic rather than abuse: one analyse call fetches up to MAX_TILES
images from a third-party provider and then spends ~11 s of CPU, so a few
concurrent callers exhaust the tile provider's goodwill and the box's cores
together — and the tile ToS is not ours to spend.

The behaviour that matters and is easy to get wrong: it must be **off by
default** (so localhost development is untouched), it must key on the forwarded
client rather than the proxy (or everyone shares one bucket), and a malformed
setting must not silently become "no limit" without saying so.
"""

import importlib

import pytest


@pytest.fixture
def app_mod(monkeypatch):
    """Reload webapp.app so RSOLAR_RATE_LIMIT is read afresh."""
    def _load(spec: str | None):
        if spec is None:
            monkeypatch.delenv("RSOLAR_RATE_LIMIT", raising=False)
        else:
            monkeypatch.setenv("RSOLAR_RATE_LIMIT", spec)
        import webapp.app as m
        importlib.reload(m)
        m._RATE_HITS.clear()
        return m
    yield _load
    # Leave the module in its default (disabled) state for other tests.
    monkeypatch.delenv("RSOLAR_RATE_LIMIT", raising=False)
    import webapp.app as m
    importlib.reload(m)


class _Req:
    """Minimal stand-in for a starlette Request."""

    def __init__(self, host="1.2.3.4", forwarded=None):
        self.headers = {"x-forwarded-for": forwarded} if forwarded else {}
        self.client = type("C", (), {"host": host})()


def test_parse_rate_limit_forms(app_mod):
    m = app_mod(None)
    assert m._parse_rate_limit("30/3600") == (30, 3600.0)
    assert m._parse_rate_limit("5/60") == (5, 60.0)
    assert m._parse_rate_limit("") is None
    assert m._parse_rate_limit("   ") is None


def test_parse_rate_limit_rejects_nonsense(app_mod):
    """A typo must not silently disable the limit without a word."""
    m = app_mod(None)
    for bad in ("abc", "0/60", "-1/60", "10/0", "10/-5", "/", "10/abc"):
        assert m._parse_rate_limit(bad) is None, bad


def test_disabled_by_default(app_mod):
    """Localhost development must be unaffected."""
    m = app_mod(None)
    assert m.RATE_LIMIT is None
    for _ in range(200):
        m._check_rate_limit(_Req())        # never raises


def test_allows_up_to_the_budget_then_429(app_mod):
    m = app_mod("3/3600")
    from fastapi import HTTPException
    for _ in range(3):
        m._check_rate_limit(_Req())
    with pytest.raises(HTTPException) as e:
        m._check_rate_limit(_Req())
    assert e.value.status_code == 429
    assert "Retry-After" in e.value.headers


def test_clients_have_separate_budgets(app_mod):
    m = app_mod("2/3600")
    from fastapi import HTTPException
    for _ in range(2):
        m._check_rate_limit(_Req(host="1.1.1.1"))
    m._check_rate_limit(_Req(host="2.2.2.2"))      # different client, fine
    with pytest.raises(HTTPException):
        m._check_rate_limit(_Req(host="1.1.1.1"))


def test_honours_x_forwarded_for(app_mod):
    """Anything public sits behind a proxy. Keying on the socket peer would put
    every user in one bucket and rate-limit the whole world together."""
    m = app_mod("2/3600")
    from fastapi import HTTPException
    for _ in range(2):
        m._check_rate_limit(_Req(host="10.0.0.1", forwarded="203.0.113.9"))
    # Same proxy, different real client: must still be allowed.
    m._check_rate_limit(_Req(host="10.0.0.1", forwarded="203.0.113.10"))
    with pytest.raises(HTTPException):
        m._check_rate_limit(_Req(host="10.0.0.1", forwarded="203.0.113.9"))


def test_forwarded_takes_first_hop(app_mod):
    m = app_mod("1/3600")
    from fastapi import HTTPException
    m._check_rate_limit(_Req(forwarded="203.0.113.9, 70.41.3.18, 150.172.238.178"))
    with pytest.raises(HTTPException):
        m._check_rate_limit(_Req(forwarded="203.0.113.9, 10.0.0.5"))


def test_window_expiry_frees_the_budget(app_mod, monkeypatch):
    m = app_mod("2/60")
    from fastapi import HTTPException
    t = [1_000_000.0]
    monkeypatch.setattr(m.time, "time", lambda: t[0])
    m._check_rate_limit(_Req())
    m._check_rate_limit(_Req())
    with pytest.raises(HTTPException):
        m._check_rate_limit(_Req())
    t[0] += 61                      # window passes
    m._check_rate_limit(_Req())     # budget is back


def test_missing_request_object_does_not_crash(app_mod):
    """The endpoints declare `request: Request = None`, so a direct call in a
    test (or any path where FastAPI does not inject it) must degrade rather
    than raise AttributeError."""
    m = app_mod("5/3600")
    m._check_rate_limit(None)
