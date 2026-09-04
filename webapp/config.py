"""Every constant the estimate depends on, in one place, with its source.

An estimate whose assumptions are visible is trustworthy; one that hides them is
not. Everything here is surfaced in the API's ``/api/assumptions`` response and
rendered in the UI's Assumptions panel, so a user can see exactly what produced
the number — and change the ones that are local to them.

Nothing in this file is a modelling constant. The model's own preprocessing
(normalisation, window, threshold) is read from the ONNX sidecar manifest —
never from a literal in the source. That separation is what stops the F-01 class
of bug coming back.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

WEBAPP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_ROOT.parent
MODELS_DIR = WEBAPP_ROOT / "models"
CACHE_DIR = Path(os.environ.get("RSOLAR_CACHE", WEBAPP_ROOT / ".cache"))

# Which exported model to serve, by file stem (no extension). Empty = pick the
# one whose sidecar says "default": true, else the newest .onnx.
#
# This exists because "newest wins" is a booby trap once more than one model is
# exported: dropping a specialised checkpoint into models/ would silently
# replace the general one for every user, and the symptom (every area estimate
# doubles) looks nothing like the cause.
SERVED_MODEL = os.environ.get("RSOLAR_MODEL", "").strip()


# --------------------------------------------------------------------------- #
# Imagery
# --------------------------------------------------------------------------- #
# Web Mercator zoom whose ground resolution matches the model's training GSD.
#   metres_per_pixel = 156543.0339 * cos(lat) / 2**z
#   z=19 -> 0.299 m/px at the equator, 0.274 at 23.26 N (Bhopal)
# Inria is 0.3 m/px, so z=19 keeps train and deploy domains aligned. Serving at
# any other zoom silently changes the effective scale the network sees.
SERVING_ZOOM = 19
TILE_PX = 256

# Esri World Imagery: global, no API key, works at z=19 in most populated areas.
# Attribution is REQUIRED and is rendered in the UI footer.
#
# Read the provider's terms before anything public-facing. Several providers —
# Google Maps Platform most strictly — prohibit running ML over their imagery or
# caching derived products. Esri's World Imagery is widely used for exactly this
# kind of non-commercial/evaluation work; Mapbox Raster Tiles is the pragmatic
# paid alternative with explicit programmatic-access terms.
TILE_PROVIDERS: dict[str, dict] = {
    "esri": {
        "label": "Esri World Imagery",
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        "attribution": ("Imagery &copy; Esri, Maxar, Earthstar Geographics, "
                        "and the GIS User Community"),
        "max_zoom": 19,
    },
    "mapbox": {
        # Needs RSOLAR_MAPBOX_TOKEN. Higher quality and clearer terms; the free
        # tier is 200k tiles/month.
        "label": "Mapbox Satellite",
        "url": ("https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg90"
                "?access_token={token}"),
        "attribution": "&copy; Mapbox &copy; Maxar",
        "max_zoom": 20,
    },
}

TILE_PROVIDER = os.environ.get("RSOLAR_TILE_PROVIDER", "esri")
MAPBOX_TOKEN = os.environ.get("RSOLAR_MAPBOX_TOKEN", "")

# Guard rails. A 1 km^2 AOI at z=19 is ~180 tiles and tens of seconds of CPU
# inference; without a cap a stray drag can request a whole city.
MAX_TILES = int(os.environ.get("RSOLAR_MAX_TILES", "256"))
TILE_FETCH_CONCURRENCY = 8
TILE_TIMEOUT_S = 20.0
USER_AGENT = "rooftop-solar-potential-detection/1.0 (research; localhost)"


# --------------------------------------------------------------------------- #
# Solar assumptions
# --------------------------------------------------------------------------- #
@dataclass
class SolarParams:
    """User-adjustable inputs to the energy and money model.

    Defaults are Indian-residential because that is the project's origin, but
    every one of them is a request parameter — the model itself is global.
    """

    # --- area -> installable area -------------------------------------------
    # The model predicts building FOOTPRINT. Usable PV area is smaller: setbacks
    # and walkways (fire code), parapet and water-tank clearance, AC units,
    # inter-row spacing for tilted arrays, and roof faces pointing the wrong way.
    # 0.70-0.80 is the standard planning range for flat roofs; 0.75 is a fair
    # default. This is the single biggest lever on the final number, which is
    # exactly why it is a slider and not a hidden constant.
    packing_factor: float = 0.75

    # --- installable area -> capacity ---------------------------------------
    # Modern mono-PERC / TOPCon modules are 20-22% efficient at STC (1000 W/m^2),
    # so ~200-220 W per m^2 of module. 0.20 is conservative.
    module_efficiency: float = 0.20

    # --- capacity -> energy --------------------------------------------------
    # PVWatts v8 default system losses: soiling 2%, shading 3%, wiring 2%,
    # connections 0.5%, light-induced degradation 1.5%, nameplate 1%, availability
    # 3% -> ~14% combined. Inverter efficiency is modelled separately by PVGIS.
    system_losses_pct: float = 14.0
    tilt_deg: float | None = None      # None -> PVGIS optimises for the latitude
    azimuth_deg: float = 0.0           # 0 = equator-facing (S in N hemisphere)

    # --- energy -> money -----------------------------------------------------
    currency: str = "INR"
    currency_symbol: str = "₹"
    tariff_per_kwh: float = 6.5        # MP domestic slab, mid-range, 2025
    cost_per_kwp: float = 45000.0      # residential rooftop, installed, pre-subsidy
    subsidy: float = 0.0               # absolute; see subsidy_note

    # --- energy -> CO2 -------------------------------------------------------
    # India CEA CO2 Baseline Database v20 (2024): combined margin ~0.71 kg/kWh.
    # EU average is ~0.23, US ~0.37 — override per country.
    grid_emission_kg_per_kwh: float = 0.71

    def to_dict(self) -> dict:
        return asdict(self)


# Where each default came from, shown in the UI so nothing is a magic number.
ASSUMPTION_SOURCES: dict[str, str] = {
    "packing_factor": ("Standard rooftop-PV planning range 0.70-0.80 for flat "
                       "roofs after setbacks, walkways, parapet/tank clearance "
                       "and inter-row spacing."),
    "module_efficiency": ("Mono-PERC/TOPCon modules are 20-22% efficient at STC; "
                          "0.20 => ~200 W per m^2 of module."),
    "system_losses_pct": "PVWatts v8 default combined system losses (14%).",
    "tilt_deg": "Blank = PVGIS picks the loss-optimal tilt for the latitude.",
    "azimuth_deg": "0 deg = equator-facing (south in the northern hemisphere).",
    "tariff_per_kwh": ("Madhya Pradesh domestic tariff, mid slab, 2025. Replace "
                       "with your own bill's per-unit rate."),
    "cost_per_kwp": ("Typical Indian residential rooftop installed cost per kWp, "
                     "pre-subsidy, 2025."),
    "subsidy": ("India's PM Surya Ghar gives Rs 30,000/kW for the first 2 kW and "
                "Rs 18,000/kW beyond, capped at Rs 78,000. Rates change - enter "
                "the current figure rather than trusting a default."),
    "grid_emission_kg_per_kwh": ("India CEA CO2 Baseline Database v20 (2024), "
                                 "combined margin ~0.71 kg/kWh. EU ~0.23, US ~0.37."),
}

# PVGIS (EU Joint Research Centre): free, no API key, global coverage including
# India. Gives location-specific yield rather than a hardcoded sun-hours number.
PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
# MRcalc: monthly *radiation* — horizontal irradiation (GHI), optimal-plane
# irradiation and mean air temperature. This is the location's solar resource,
# independent of any system design, and is what makes two sites comparable.
PVGIS_MR_URL = "https://re.jrc.ec.europa.eu/api/v5_2/MRcalc"
PVGIS_TIMEOUT_S = 25.0

# Fallback if PVGIS is unreachable (offline demo). Deliberately crude, and the
# API response is flagged so the UI can say the number is a fallback.
FALLBACK_SPECIFIC_YIELD_KWH_PER_KWP = 1400.0  # ~India average


# --------------------------------------------------------------------------- #
# Vectorisation
# --------------------------------------------------------------------------- #
# Drop specks below this many square metres — at 0.3 m/px a 10 m^2 blob is ~110
# pixels. Anything smaller is noise, not a roof worth panelling.
MIN_BUILDING_AREA_M2 = 10.0
# Douglas-Peucker tolerance as a fraction of the contour perimeter. Measured on
# a dense Bangalore block: 0.01 threw away 6.1% of the detected area by cutting
# corners off real roofs; 0.005 loses 4.6% and 0.002 only 3.8%, but vertex count
# triples. 0.005 is the knee — and since this number is multiplied straight into
# the energy estimate, biasing it low by 6% was not acceptable.
POLYGON_SIMPLIFY_FRAC = 0.005
# Morphological open/close kernel, in pixels, to clean up ragged mask edges.
MORPH_KERNEL_PX = 3


@dataclass
class AppConfig:
    serving_zoom: int = SERVING_ZOOM
    max_tiles: int = MAX_TILES
    tile_provider: str = TILE_PROVIDER
    solar: SolarParams = field(default_factory=SolarParams)


def tile_provider_config(name: str | None = None) -> dict:
    """Resolve a provider entry, substituting the Mapbox token if needed."""
    key = (name or TILE_PROVIDER).lower()
    if key not in TILE_PROVIDERS:
        raise ValueError(f"unknown tile provider {key!r}. "
                         f"Known: {sorted(TILE_PROVIDERS)}")
    cfg = dict(TILE_PROVIDERS[key])
    if key == "mapbox":
        if not MAPBOX_TOKEN:
            raise ValueError(
                "tile provider 'mapbox' needs RSOLAR_MAPBOX_TOKEN in the "
                "environment. Unset RSOLAR_TILE_PROVIDER to use Esri instead.")
        cfg["url"] = cfg["url"].replace("{token}", MAPBOX_TOKEN)
    cfg["name"] = key
    return cfg
