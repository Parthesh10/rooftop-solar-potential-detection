"""How far is this AOI from anything the model has actually seen?

The model was trained on five cities. It is very good on imagery that looks like
them and measurably worse elsewhere — verified by eye on 2026-09-03: downtown
Austin (a training city) segments cleanly, while Bhopal misses a substantial
fraction of buildings.

A user analysing their own roof has no way to know that. Silently returning a
confident-looking number for an untested region would be the dishonest choice,
so the API attaches a plain-language note saying where the estimate sits
relative to the training data.

Distance is a crude proxy for "does the built environment look like the training
set" — it is not the real variable. But it is honest, cheap, and catches the case
that matters: someone in a region the model has never seen.
"""

from __future__ import annotations

import math

__all__ = ["TRAINING_CITIES", "coverage_note"]

# The five Inria training cities, with the region each one stands in for.
TRAINING_CITIES: list[tuple[str, float, float]] = [
    ("Austin, TX",            30.27, -97.74),
    ("Chicago, IL",           41.88, -87.63),
    ("Kitsap County, WA",     47.60, -122.65),
    ("Innsbruck / Tyrol, AT", 47.27, 11.39),
    ("Vienna, AT",            48.21, 16.37),
]

# Beyond this, call the region untested rather than implying it was covered.
NEAR_KM = 150.0
REGIONAL_KM = 900.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def coverage_note(lat: float, lon: float) -> dict:
    """Classify an AOI against the training footprint.

    Returns ``{level, nearest, distance_km, note}`` where ``level`` is one of
    ``"trained"``, ``"regional"``, ``"untested"``.
    """
    name, dist = min(
        ((n, _haversine_km(lat, lon, clat, clon)) for n, clat, clon in TRAINING_CITIES),
        key=lambda t: t[1],
    )

    if dist <= NEAR_KM:
        level = "trained"
        note = (f"This area is close to {name}, one of the model's training "
                f"cities. Detection quality here should match the reported "
                f"accuracy.")
    elif dist <= REGIONAL_KM:
        level = "regional"
        note = (f"The nearest training city is {name}, about {dist:,.0f} km "
                f"away. Rooftops in the same region usually look similar, so "
                f"results are typically reliable — but this exact area was not "
                f"in the training set.")
    else:
        level = "untested"
        note = (f"The model was trained on five cities in the USA and Austria; "
                f"the nearest ({name}) is about {dist:,.0f} km from here. It "
                f"has never been evaluated on rooftops in this region and is "
                f"known to miss buildings whose style, density or materials "
                f"differ from Western suburban and Alpine-European housing. "
                f"Treat this estimate as indicative, and expect the detected "
                f"roof area to be an under-count.")

    return {"level": level, "nearest": name,
            "distance_km": round(dist, 0), "note": note}


# --------------------------------------------------------------------------- #
# Threshold policy
# --------------------------------------------------------------------------- #
# The optimal decision threshold is not a property of the model alone — it is a
# property of the model *on a domain*. Measured on Inria val (in-distribution),
# 0.50 and 0.65 differ by 0.002 IoU: noise. Measured on Bangalore
# (out-of-distribution), going from 0.50 to 0.60 lost 6-11% of detected roof
# area, because the model is systematically under-confident on rooftops that do
# not look like its training set and a stricter cut removes real buildings.
#
# So: keep 0.50 as the in-distribution default (near-optimal there, and safe),
# and relax it where the model is known to be under-confident. This is a
# reasoned correction for a measured under-confidence, **not** a threshold
# tuned on labelled local data — there is no labelled Indian set to tune on.
# The UI says so, and the slider overrides it.
THRESHOLD_BY_LEVEL: dict[str, float] = {
    "trained": 0.50,
    "regional": 0.45,
    "untested": 0.40,
}


def suggested_threshold(level: str, default: float = 0.50) -> float:
    """Decision threshold to use for an AOI at this coverage level."""
    return THRESHOLD_BY_LEVEL.get(level, default)


def threshold_note(level: str, threshold: float) -> str | None:
    """Plain-language reason for a non-default threshold, or None."""
    if level == "untested":
        return (f"Detection sensitivity was raised (threshold {threshold:.2f} "
                f"instead of 0.50) because the model is under-confident on "
                f"rooftops unlike its training data and would otherwise miss "
                f"buildings here. If you see areas marked that are not roofs, "
                f"raise the threshold under Detection sensitivity.")
    if level == "regional":
        return (f"Detection sensitivity was raised slightly (threshold "
                f"{threshold:.2f}) as this area sits outside the training "
                f"cities.")
    return None
