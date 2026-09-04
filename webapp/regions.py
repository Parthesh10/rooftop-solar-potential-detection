"""Which numbers are right *here*?

The app was born Indian, so every default in :mod:`webapp.config` is Indian: a
Madhya Pradesh tariff, a rupee symbol, the CEA grid emission factor. That is
fine for Bhopal and wrong everywhere else, and "wrong everywhere else" is not
good enough for a model whose whole selling point is that it detects roofs
anywhere.

This module is the *tabulated* half of the fix. It answers the question **what
is known about this place before we look at a single pixel?** — and it is
deliberately narrow, because only a few things about a location can be looked up
from a table without lying:

* **Money and grid.** Currency, a representative residential tariff, an
  installed cost per kWp, the grid's CO2 intensity. These really are regional
  constants, they change slowly, and being within 30% is far better than being
  in the wrong currency.
* **A plausible band for the decision threshold.** Not the threshold itself —
  see below — but the range outside which any calibration result is a bug.
* **The built form to expect.** Whether roofs here are 40 m^2 houses packed wall
  to wall or 2000 m^2 warehouses, which is what decides the morphological
  kernel.

What this module deliberately does **not** do is tabulate a threshold per city.
You cannot tabulate the world: there are tens of thousands of cities and the
built form changes across a single one. The threshold is *measured* per AOI
instead — see :mod:`webapp.calibration`. Tabulate what is genuinely regional;
measure what is local.

Adding a region is a dict literal with a bounding box. Entries merge
broadest-first, so a country inherits from ``GLOBAL`` and a sub-region inherits
from its country without repeating anything.

**Bounding boxes are rectangles and countries are not.** India's box contains
Sri Lanka, Nepal, Bhutan and most of Bangladesh; Sri Lanka is carved back out
because it is cleanly separable, and the others are not because any rectangle
drawn around them also swallows real Indian territory. The consequence is
bounded and visible: those neighbours inherit India's *economics* labels, while
their detection settings — the only thing that changes what the model does — are
identical either way, since the whole of South Asia shares one threshold band
and one built-form expectation. ``matched`` in the API response lists every
region that fired, so a wrong match is legible rather than silent. A reverse
geocoder would fix it properly and is the obvious upgrade if this table ever
carries numbers worth being precise about.

**Every economic figure here is indicative and user-editable.** They are
starting points that keep a first estimate in the right currency and the right
order of magnitude, not authoritative rates. ``economics_confidence`` says how
much to trust each set, and the UI shows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Region", "RegionProfile", "REGIONS", "resolve", "region_keys"]


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Region:
    """One entry. ``bbox`` is ``(west, south, east, north)``; None = everywhere."""

    key: str
    name: str
    bbox: tuple[float, float, float, float] | None
    values: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)

    def contains(self, lat: float, lon: float) -> bool:
        if self.bbox is None:
            return True
        w, s, e, n = self.bbox
        return w <= lon <= e and s <= lat <= n

    @property
    def span(self) -> float:
        """Bbox area in square degrees — the specificity rank. None = infinite."""
        if self.bbox is None:
            return float("inf")
        w, s, e, n = self.bbox
        return (e - w) * (n - s)


# Every field a profile can carry, with the value used when nothing more
# specific matches. The global fallback is deliberately currency-neutral and
# flagged as having no regional data: showing a Brazilian user rupees would be
# worse than showing a placeholder that admits it is one.
GLOBAL = Region(
    key="global",
    name="Unlisted region",
    bbox=None,
    values={
        # --- detection ------------------------------------------------------
        # The band any calibrated threshold is clamped into. Wide, because the
        # whole point of calibrating is that the right value is not known in
        # advance; narrow enough that a broken calibration cannot mask the
        # entire image or none of it.
        "threshold_band": (0.30, 0.65),
        # Median rooftop footprint to expect, m^2. Drives the morphological
        # kernel: small dense housing needs a smaller kernel or adjacent houses
        # close into one blob.
        "typical_building_m2": 120.0,
        # --- money and grid --------------------------------------------------
        "currency": "USD",
        "currency_symbol": "$",
        "tariff_per_kwh": 0.15,
        "cost_per_kwp": 1500.0,
        "grid_emission_kg_per_kwh": 0.45,
        "packing_factor": 0.75,
        "subsidy_note": "",
        "economics_confidence": "none",
    },
    sources={
        "tariff_per_kwh": "Placeholder. No regional data for this location — "
                          "replace with your own bill's per-unit rate.",
        "cost_per_kwp": "Placeholder global average installed cost.",
        "grid_emission_kg_per_kwh": "Placeholder; world average grid intensity "
                                    "is roughly 0.45 kg CO2/kWh.",
        "packing_factor": "Standard rooftop-PV planning range 0.70-0.80 for "
                          "flat roofs after setbacks, walkways and clearances.",
    },
)


REGIONS: list[Region] = [
    GLOBAL,

    # ------------------------------------------------------------------ India #
    # The project's home region and the one the model is measurably worst on,
    # so it gets the widest downward room in the threshold band.
    Region(
        key="in",
        name="India",
        bbox=(68.0, 6.5, 97.5, 35.7),
        values={
            "threshold_band": (0.28, 0.60),
            # Indian urban housing is small-plot: 30-60 m^2 rooftops packed with
            # no gap is the norm in the residential blocks this app gets pointed
            # at. Verified by eye on Bangalore and Bhopal, 2026-09.
            "typical_building_m2": 60.0,
            "currency": "INR",
            "currency_symbol": "₹",
            "tariff_per_kwh": 6.5,
            "cost_per_kwp": 45000.0,
            "grid_emission_kg_per_kwh": 0.71,
            # Lower than the 0.75 global default: Indian roofs carry water
            # tanks, stair headrooms and parapets on almost every house, and the
            # roofs are small enough that those eat a larger fraction.
            "packing_factor": 0.70,
            "subsidy_note": ("PM Surya Ghar: Rs 30,000/kW for the first 2 kW "
                             "and Rs 18,000/kW beyond, capped at Rs 78,000. "
                             "Rates change — enter the current figure."),
            "economics_confidence": "medium",
        },
        sources={
            "tariff_per_kwh": "Madhya Pradesh domestic tariff, mid slab, 2025. "
                              "Indian tariffs vary widely by state and slab — "
                              "use your own bill.",
            "cost_per_kwp": "Typical Indian residential rooftop installed cost "
                            "per kWp, pre-subsidy, 2025.",
            "grid_emission_kg_per_kwh": "India CEA CO2 Baseline Database v20 "
                                        "(2024), combined margin ~0.71 kg/kWh.",
            "packing_factor": "Reduced from the 0.75 flat-roof planning default "
                              "for water tanks, stair headrooms and parapets on "
                              "small Indian rooftops.",
            "typical_building_m2": "Observed on Bangalore and Bhopal imagery, "
                                   "2026-09.",
        },
    ),

    # ------------------------------------------------------- the model's turf #
    Region(
        key="us",
        name="United States",
        bbox=(-125.0, 24.5, -66.9, 49.4),
        values={
            # Three of the five training cities are here, so the model is at its
            # best and the band stays tight around the in-distribution 0.50.
            "threshold_band": (0.40, 0.70),
            "typical_building_m2": 180.0,
            "currency": "USD",
            "currency_symbol": "$",
            "tariff_per_kwh": 0.16,
            "cost_per_kwp": 2800.0,
            "grid_emission_kg_per_kwh": 0.37,
            "packing_factor": 0.75,
            "subsidy_note": ("Federal residential clean-energy credit plus "
                             "state and utility incentives — enter the net "
                             "figure that applies to you."),
            "economics_confidence": "medium",
        },
        sources={
            "tariff_per_kwh": "US residential average, EIA, ~2024. The "
                              "state spread is roughly $0.11-$0.32.",
            "cost_per_kwp": "NREL benchmark residential installed cost, ~2024.",
            "grid_emission_kg_per_kwh": "US grid average, ~0.37 kg CO2/kWh.",
        },
    ),
    Region(
        key="ca",
        name="Canada",
        bbox=(-141.0, 41.7, -52.6, 70.0),
        values={
            "threshold_band": (0.40, 0.70),
            "typical_building_m2": 180.0,
            "currency": "CAD",
            "currency_symbol": "$",
            "tariff_per_kwh": 0.14,
            "cost_per_kwp": 3000.0,
            # Hydro-dominated: a Canadian kWh displaces far less CO2 than an
            # Indian or Australian one, and pretending otherwise inflates the
            # headline climate number several-fold.
            "grid_emission_kg_per_kwh": 0.13,
            "economics_confidence": "low",
        },
        sources={
            "grid_emission_kg_per_kwh": "Canada's grid is largely hydro and "
                                        "nuclear; the national average sits "
                                        "around 0.12-0.15 kg CO2/kWh.",
        },
    ),
    Region(
        key="eu",
        name="Europe",
        bbox=(-11.0, 35.0, 32.0, 71.5),
        values={
            "threshold_band": (0.40, 0.70),
            "typical_building_m2": 150.0,
            "currency": "EUR",
            "currency_symbol": "€",
            "tariff_per_kwh": 0.28,
            "cost_per_kwp": 1600.0,
            "grid_emission_kg_per_kwh": 0.23,
            "economics_confidence": "low",
        },
        sources={
            "tariff_per_kwh": "EU household electricity price band, ~2024. The "
                              "member-state spread is very wide.",
            "grid_emission_kg_per_kwh": "EU-27 grid average, ~0.23 kg CO2/kWh.",
        },
    ),
    Region(
        key="at",
        name="Austria",
        bbox=(9.5, 46.3, 17.2, 49.1),
        values={
            # Innsbruck and Vienna are both training cities.
            "threshold_band": (0.42, 0.72),
            "grid_emission_kg_per_kwh": 0.15,
            "economics_confidence": "low",
        },
        sources={
            "grid_emission_kg_per_kwh": "Austria's grid is hydro-heavy and well "
                                        "below the EU average.",
        },
    ),
    Region(
        key="de",
        name="Germany",
        bbox=(5.8, 47.2, 15.1, 55.1),
        values={
            "tariff_per_kwh": 0.35,
            "grid_emission_kg_per_kwh": 0.38,
            "economics_confidence": "low",
        },
        sources={"tariff_per_kwh": "German household electricity price, ~2024."},
    ),
    Region(
        key="gb",
        name="United Kingdom",
        bbox=(-8.7, 49.8, 2.1, 61.0),
        values={
            "currency": "GBP",
            "currency_symbol": "£",
            "tariff_per_kwh": 0.25,
            "cost_per_kwp": 1700.0,
            "grid_emission_kg_per_kwh": 0.21,
            "economics_confidence": "low",
        },
        sources={"grid_emission_kg_per_kwh": "UK grid average, ~2024."},
    ),
    Region(
        key="au",
        name="Australia",
        bbox=(112.0, -44.0, 154.0, -9.0),
        values={
            "threshold_band": (0.35, 0.65),
            "typical_building_m2": 180.0,
            "currency": "AUD",
            "currency_symbol": "$",
            "tariff_per_kwh": 0.30,
            "cost_per_kwp": 1100.0,
            "grid_emission_kg_per_kwh": 0.63,
            "economics_confidence": "low",
        },
        sources={
            "cost_per_kwp": "Australia has the world's cheapest installed "
                            "residential rooftop PV.",
        },
    ),

    # -------------------------------------------- neighbours worth splitting #
    # These share India's built form but not its economics, so they inherit the
    # threshold band and the small-footprint expectation and nothing else.
    Region(
        key="sa",
        name="South Asia",
        bbox=(60.0, 5.0, 92.5, 38.5),
        values={
            "threshold_band": (0.28, 0.60),
            "typical_building_m2": 60.0,
            "packing_factor": 0.70,
            "economics_confidence": "none",
        },
        sources={"typical_building_m2": "Same small-plot urban form as India."},
    ),
    # Cleanly separable from India's rectangle — an island, with open water
    # between it and the nearest Indian land. Its neighbours are not, so they
    # are left inheriting India's economics; see the module docstring.
    Region(
        key="lk",
        name="Sri Lanka",
        bbox=(79.6, 5.8, 82.0, 10.0),
        values={
            # Same small-plot built form as India, but no economic data — and a
            # placeholder that admits it beats confidently wrong rupees.
            "currency": "USD",
            "currency_symbol": "$",
            "tariff_per_kwh": 0.15,
            "cost_per_kwp": 1500.0,
            "grid_emission_kg_per_kwh": 0.45,
            "subsidy_note": "",
            "economics_confidence": "none",
        },
        sources={
            "tariff_per_kwh": "No regional data for Sri Lanka — this is a "
                              "placeholder. Enter your own per-unit rate.",
        },
    ),
    Region(
        key="sea",
        name="Southeast Asia",
        bbox=(92.0, -11.0, 141.0, 24.0),
        values={
            "threshold_band": (0.28, 0.60),
            "typical_building_m2": 70.0,
            "packing_factor": 0.72,
            "economics_confidence": "none",
        },
    ),
    Region(
        key="af",
        name="Sub-Saharan Africa",
        bbox=(-18.0, -35.0, 52.0, 16.0),
        values={
            "threshold_band": (0.28, 0.60),
            "typical_building_m2": 55.0,
            "packing_factor": 0.72,
            "economics_confidence": "none",
        },
        sources={
            "typical_building_m2": "Small-footprint housing dominates; the "
                                   "Google Open Buildings footprint "
                                   "distribution for the region peaks well "
                                   "under 100 m^2.",
        },
    ),
]


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegionProfile:
    """What the table knows about one point on the globe."""

    key: str
    name: str
    matched: list[str]                # every region that matched, broad -> narrow
    threshold_band: tuple[float, float]
    typical_building_m2: float
    currency: str
    currency_symbol: str
    tariff_per_kwh: float
    cost_per_kwp: float
    grid_emission_kg_per_kwh: float
    packing_factor: float
    subsidy_note: str
    economics_confidence: str         # "none" | "low" | "medium"
    sources: dict[str, str]

    def economics(self) -> dict:
        """The subset the UI pre-fills into the Assumptions panel."""
        return {
            "currency": self.currency,
            "currency_symbol": self.currency_symbol,
            "tariff_per_kwh": self.tariff_per_kwh,
            "cost_per_kwp": self.cost_per_kwp,
            "grid_emission_kg_per_kwh": self.grid_emission_kg_per_kwh,
            "packing_factor": self.packing_factor,
        }

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "matched": self.matched,
            "threshold_band": list(self.threshold_band),
            "typical_building_m2": self.typical_building_m2,
            "economics": self.economics(),
            "economics_confidence": self.economics_confidence,
            "subsidy_note": self.subsidy_note,
            "sources": self.sources,
        }


def resolve(lat: float, lon: float) -> RegionProfile:
    """Merge every matching region, broadest first, so the narrowest wins."""
    matches = sorted((r for r in REGIONS if r.contains(lat, lon)),
                     key=lambda r: -r.span)
    if not matches:                    # GLOBAL has bbox None, so unreachable
        matches = [GLOBAL]

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for region in matches:
        values.update(region.values)
        sources.update(region.sources)

    lo, hi = (float(x) for x in values["threshold_band"])
    narrowest = matches[-1]
    return RegionProfile(
        key=narrowest.key,
        name=narrowest.name,
        matched=[r.key for r in matches],
        threshold_band=(lo, hi),
        typical_building_m2=float(values["typical_building_m2"]),
        currency=str(values["currency"]),
        currency_symbol=str(values["currency_symbol"]),
        tariff_per_kwh=float(values["tariff_per_kwh"]),
        cost_per_kwp=float(values["cost_per_kwp"]),
        grid_emission_kg_per_kwh=float(values["grid_emission_kg_per_kwh"]),
        packing_factor=float(values["packing_factor"]),
        subsidy_note=str(values.get("subsidy_note", "")),
        economics_confidence=str(values["economics_confidence"]),
        sources=sources,
    )


def region_keys() -> list[str]:
    return [r.key for r in REGIONS]
