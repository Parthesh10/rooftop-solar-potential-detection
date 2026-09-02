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
