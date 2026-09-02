"""Roof area -> installable capacity -> energy -> money and CO2.

The chain, with the honest name for each step:

    detected roof footprint (m^2)          <- what the model actually predicts
      x packing factor (0.70-0.80)         <- setbacks, walkways, parapets, tanks,
                                              inter-row spacing, bad azimuths
      = usable PV area (m^2)               <- an ESTIMATE, never a site survey
      x module efficiency (~0.20)          <- ~200 W per m^2
      = capacity (kWp)
      -> PVGIS(lat, lon, kWp, tilt, loss)  <- location-specific annual yield
      = annual kWh
      x tariff                             = annual savings
      x grid emission factor               = CO2 avoided

Irradiance is **not** hardcoded. PVGIS (EU Joint Research Centre) is free, needs
no API key, and covers the whole world including India. If it is unreachable the
result is still returned but flagged ``"source": "fallback"`` so the UI can say
so rather than quietly presenting a guess as data.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

from webapp.config import (
    CACHE_DIR,
    FALLBACK_SPECIFIC_YIELD_KWH_PER_KWP,
    PVGIS_TIMEOUT_S,
    PVGIS_URL,
    SolarParams,
)

__all__ = ["estimate", "pvgis_yield", "MONTHS"]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_cache_lock = threading.Lock()


def _cache_path(lat: float, lon: float, tilt, azimuth: float, loss: float) -> Path:
    # Round to ~1 km: irradiance does not change meaningfully below that, and it
    # keeps the cache from exploding over slightly different AOI centroids.
    key = f"{lat:.2f}_{lon:.2f}_{tilt}_{azimuth:.0f}_{loss:.0f}"
    return CACHE_DIR / "pvgis" / f"{key}.json"


def pvgis_yield(lat: float, lon: float, params: SolarParams) -> dict:
    """Annual and monthly specific yield (kWh per kWp installed) at this site.

    Queried for a nominal 1 kWp so the result scales linearly and one cache entry
    serves every capacity at that location.
    """
    tilt = params.tilt_deg
    path = _cache_path(lat, lon, tilt if tilt is not None else "opt",
                       params.azimuth_deg, params.system_losses_pct)

    with _cache_lock:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass  # corrupt cache entry: refetch

    query = {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "peakpower": 1.0,
        "loss": params.system_losses_pct,
        "outputformat": "json",
        "pvtechchoice": "crystSi",
        "mountingplace": "building",
        "angle": tilt if tilt is not None else 0,
        "aspect": params.azimuth_deg,
    }
    if tilt is None:
        query["optimalangles"] = 1

    try:
        import httpx

        r = httpx.get(PVGIS_URL, params=query, timeout=PVGIS_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
        totals = data["outputs"]["totals"]["fixed"]
        monthly = [m["E_m"] for m in data["outputs"]["monthly"]["fixed"]]
        inputs = data.get("inputs", {}).get("mounting_system", {}).get("fixed", {})
        out = {
            "annual_kwh_per_kwp": float(totals["E_y"]),
            "monthly_kwh_per_kwp": [float(v) for v in monthly],
            "optimal_tilt_deg": float(inputs.get("slope", {}).get("value", tilt or 0)),
            "azimuth_deg": float(inputs.get("azimuth", {}).get("value",
                                                               params.azimuth_deg)),
            "source": "PVGIS v5.2 (EU JRC), SARAH3 / ERA5",
            "ok": True,
        }
    except Exception as exc:
        # Seasonal shape approximating a mid-latitude northern-hemisphere year.
        # Crude on purpose — it is flagged, not disguised.
        shape = [0.075, 0.080, 0.092, 0.094, 0.096, 0.082,
                 0.072, 0.072, 0.081, 0.086, 0.081, 0.089]
        annual = FALLBACK_SPECIFIC_YIELD_KWH_PER_KWP
        out = {
            "annual_kwh_per_kwp": annual,
            "monthly_kwh_per_kwp": [round(annual * s, 1) for s in shape],
            "optimal_tilt_deg": abs(lat) * 0.76 + 3.1,  # classic rule of thumb
            "azimuth_deg": params.azimuth_deg,
            "source": f"fallback constant ({annual:g} kWh/kWp) — PVGIS unreachable",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if out["ok"]:
        with _cache_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out), encoding="utf-8")
    return out


def estimate(roof_area_m2: float, lat: float, lon: float,
             params: SolarParams) -> dict:
    """The full chain, returned with every intermediate value exposed."""
    usable_m2 = roof_area_m2 * params.packing_factor
    # 1 kW/m^2 at STC, so kWp = area x efficiency x 1 kW/m^2.
    capacity_kwp = usable_m2 * params.module_efficiency

    yields = pvgis_yield(lat, lon, params)
    annual_kwh = capacity_kwp * yields["annual_kwh_per_kwp"]
    monthly_kwh = [capacity_kwp * v for v in yields["monthly_kwh_per_kwp"]]

    annual_savings = annual_kwh * params.tariff_per_kwh
    gross_cost = capacity_kwp * params.cost_per_kwp
    net_cost = max(gross_cost - params.subsidy, 0.0)
    payback_years = (net_cost / annual_savings) if annual_savings > 0 else None
    co2_kg = annual_kwh * params.grid_emission_kg_per_kwh

    # 25-year net benefit: standard module performance warranty period. No
    # discount rate applied and none implied — this is a gross figure.
    lifetime_years = 25
    lifetime_savings = annual_savings * lifetime_years

    return {
        "roof_area_m2": round(roof_area_m2, 1),
        "packing_factor": params.packing_factor,
        "usable_area_m2": round(usable_m2, 1),
        "capacity_kwp": round(capacity_kwp, 2),
        "annual_kwh": round(annual_kwh, 0),
        "monthly_kwh": [round(v, 0) for v in monthly_kwh],
        "specific_yield_kwh_per_kwp": round(yields["annual_kwh_per_kwp"], 0),
        "optimal_tilt_deg": round(yields["optimal_tilt_deg"], 1),
        "azimuth_deg": round(yields["azimuth_deg"], 1),
        "irradiance_source": yields["source"],
        "irradiance_ok": yields["ok"],
        "currency": params.currency,
        "currency_symbol": params.currency_symbol,
        "annual_savings": round(annual_savings, 0),
        "gross_cost": round(gross_cost, 0),
        "subsidy": round(params.subsidy, 0),
        "net_cost": round(net_cost, 0),
        "payback_years": round(payback_years, 1) if payback_years else None,
        "lifetime_years": lifetime_years,
        "lifetime_savings": round(lifetime_savings, 0),
        "co2_avoided_kg_per_year": round(co2_kg, 0),
        "co2_avoided_t_over_lifetime": round(co2_kg * lifetime_years / 1000.0, 1),
        # A tangible equivalence: ~21 kg CO2 sequestered per mature tree per year.
        "trees_equivalent": round(co2_kg / 21.0),
    }


def sanity_check_capacity(capacity_kwp: float) -> list[str]:
    """Warnings worth surfacing rather than silently returning a silly number."""
    notes = []
    if capacity_kwp > 1000:
        notes.append("Over 1 MWp — this is a utility-scale figure. Check the AOI "
                     "covers only the roofs you meant to include.")
    if 0 < capacity_kwp < 1:
        notes.append("Under 1 kWp — smaller than a typical residential system. "
                     "The detected roof may be partial.")
    return notes


def format_number(value: float, currency_symbol: str = "") -> str:
    """Indian-style grouping (1,23,456) when the symbol is a rupee, else Western."""
    if value is None:
        return "—"
    n = int(round(value))
    if currency_symbol != "₹":
        return f"{n:,}"
    s = str(abs(n))
    if len(s) <= 3:
        grouped = s
    else:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        grouped = ",".join(parts) + "," + tail
    return ("-" if n < 0 else "") + grouped
