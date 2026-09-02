"""The solar chain and the geographic-coverage honesty check.

Every number the UI shows comes through ``solar.estimate``. These tests pin the
arithmetic so a refactor cannot quietly change what a user is told their roof is
worth. PVGIS is stubbed — the tests must not depend on a network call.
"""

import pytest

from webapp import coverage, solar
from webapp.config import SolarParams


@pytest.fixture
def offline(monkeypatch):
    """Deterministic 1500 kWh/kWp site, no network."""
    monkeypatch.setattr(solar, "pvgis_yield", lambda lat, lon, p: {
        "annual_kwh_per_kwp": 1500.0,
        "monthly_kwh_per_kwp": [125.0] * 12,
        "optimal_tilt_deg": 20.0,
        "azimuth_deg": 0.0,
        "source": "stub",
        "ok": True,
    })


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #
def test_the_full_chain_is_arithmetically_what_it_claims(offline):
    p = SolarParams(packing_factor=0.75, module_efficiency=0.20,
                    tariff_per_kwh=6.5, cost_per_kwp=45000.0,
                    grid_emission_kg_per_kwh=0.71)
    r = solar.estimate(1000.0, 23.2, 77.4, p)

    assert r["usable_area_m2"] == pytest.approx(750.0)      # 1000 x 0.75
    assert r["capacity_kwp"] == pytest.approx(150.0)        # 750 x 0.20
    assert r["annual_kwh"] == pytest.approx(225000.0)       # 150 x 1500
    assert r["annual_savings"] == pytest.approx(225000 * 6.5)
    assert r["gross_cost"] == pytest.approx(150.0 * 45000)
    assert r["co2_avoided_kg_per_year"] == pytest.approx(225000 * 0.71)


def test_payback_uses_net_cost_after_subsidy(offline):
    base = SolarParams(cost_per_kwp=45000.0, subsidy=0.0)
    subsidised = SolarParams(cost_per_kwp=45000.0, subsidy=78000.0)
    a = solar.estimate(100.0, 23.2, 77.4, base)
    b = solar.estimate(100.0, 23.2, 77.4, subsidised)

    assert b["net_cost"] == pytest.approx(a["gross_cost"] - 78000.0)
    assert b["payback_years"] < a["payback_years"]


def test_subsidy_cannot_drive_net_cost_negative(offline):
    p = SolarParams(cost_per_kwp=1000.0, subsidy=10_000_000.0)
    r = solar.estimate(10.0, 23.2, 77.4, p)
    assert r["net_cost"] == 0.0


def test_zero_tariff_gives_no_payback_rather_than_dividing_by_zero(offline):
    r = solar.estimate(100.0, 23.2, 77.4, SolarParams(tariff_per_kwh=0.0))
    assert r["payback_years"] is None


def test_zero_roof_area_is_handled(offline):
    r = solar.estimate(0.0, 23.2, 77.4, SolarParams())
    assert r["capacity_kwp"] == 0.0
    assert r["annual_kwh"] == 0.0
    assert r["payback_years"] is None


def test_packing_factor_is_linear_and_dominant(offline):
    lo = solar.estimate(1000.0, 23.2, 77.4, SolarParams(packing_factor=0.5))
    hi = solar.estimate(1000.0, 23.2, 77.4, SolarParams(packing_factor=1.0))
    assert hi["annual_kwh"] == pytest.approx(2 * lo["annual_kwh"])


def test_monthly_sums_to_annual(offline):
    r = solar.estimate(500.0, 23.2, 77.4, SolarParams())
    assert sum(r["monthly_kwh"]) == pytest.approx(r["annual_kwh"], rel=0.02)


def test_lifetime_figures_are_consistent(offline):
    r = solar.estimate(500.0, 23.2, 77.4, SolarParams())
    assert r["lifetime_savings"] == pytest.approx(
        r["annual_savings"] * r["lifetime_years"], rel=1e-6)
    assert r["co2_avoided_t_over_lifetime"] == pytest.approx(
        r["co2_avoided_kg_per_year"] * r["lifetime_years"] / 1000, rel=1e-3)


def test_pvgis_failure_is_flagged_not_hidden(monkeypatch):
    """An offline fallback must be visible in the response, never disguised."""
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("httpx.get", boom)

    out = solar.pvgis_yield(23.2, 77.4, SolarParams())
    assert out["ok"] is False
    assert "fallback" in out["source"].lower()
    assert len(out["monthly_kwh_per_kwp"]) == 12
    assert out["annual_kwh_per_kwp"] > 0


def test_capacity_sanity_warnings():
    assert solar.sanity_check_capacity(5000) != []
    assert solar.sanity_check_capacity(0.4) != []
    assert solar.sanity_check_capacity(6.0) == []


def test_indian_digit_grouping():
    assert solar.format_number(123456, "₹") == "1,23,456"
    assert solar.format_number(1234567, "₹") == "12,34,567"
    assert solar.format_number(999, "₹") == "999"
    assert solar.format_number(123456, "$") == "123,456"


# --------------------------------------------------------------------------- #
# Coverage honesty
# --------------------------------------------------------------------------- #
def test_a_training_city_is_reported_as_trained():
    c = coverage.coverage_note(30.27, -97.74)     # Austin
    assert c["level"] == "trained"
    assert c["distance_km"] < 20


def test_india_is_reported_as_untested():
    c = coverage.coverage_note(23.26, 77.41)      # Bhopal
    assert c["level"] == "untested"
    assert "never been evaluated" in c["note"]
    assert c["distance_km"] > 4000


def test_a_city_between_training_sites_counts_as_trained():
    """Munich sits ~100 km from Innsbruck — same alpine-European building stock."""
    c = coverage.coverage_note(48.14, 11.58)
    assert c["level"] == "trained"


def test_further_europe_is_regional_not_trained():
    c = coverage.coverage_note(52.52, 13.40)      # Berlin, ~520 km from Vienna
    assert c["level"] == "regional"
    assert "was not in the training set" in c["note"]


@pytest.mark.parametrize("lat,lon", [(0, 0), (-33.9, 151.2), (64.1, -21.9), (23.2, 77.4)])
def test_coverage_always_returns_a_usable_note(lat, lon):
    c = coverage.coverage_note(lat, lon)
    assert c["level"] in {"trained", "regional", "untested"}
    assert c["note"] and c["nearest"]
    assert c["distance_km"] >= 0
