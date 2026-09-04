# Web app

Draw a box on a map → the model finds the rooftops → you get an estimate of how
much solar those roofs could produce, in kWh, money and CO₂.

One FastAPI process serves both the API and the UI. **No Node, no npm, no build
step.**

## Run it

```powershell
cd "E:\Claude Workspace\Rooftop Solar Detection\rooftop-solar-potential-detection"

# once: export the trained model to ONNX (writes webapp/models/)
.\.venv\Scripts\python.exe scripts\export_onnx.py

# then, any time
.\.venv\Scripts\python.exe -m webapp
```

Opens <http://127.0.0.1:8000> automatically. `--port 8080`, `--host 0.0.0.0`,
`--no-open` and `--reload` all work.

## Using it

1. **Choose an area** — search for a place, or pan the map. Then *Draw area* and
   drag a box over the roofs you care about. The read-out shows the size and how
   many imagery tiles it needs; the cap is 256 (about 1.5 km²).
2. **Assumptions** — the packing factor is the one that matters most. The model
   finds roof *outline*; only part of that takes panels. 0.75 is a fair default.
3. **Detect rooftops** — 10–30 s for a typical block. Roofs appear on the map,
   the estimate opens on the right.

Export the result as GeoJSON (opens in QGIS) or CSV.

## What it does under the hood

```
AOI bounds
  → tile list at Web Mercator z=19        tiles.py
  → fetch + stitch to one mosaic          tiles.py     (8 parallel requests)
  → sliding-window inference, 512/256     inference.py (ONNX Runtime)
  → Hann-blended probability map
  → threshold → morphological open/close  geometry.py
  → cv2.findContours + Douglas-Peucker    geometry.py
  → rings to lon/lat, geodesic area       geometry.py  (pyproj.Geod)
  → area → kWp → PVGIS → kWh → ₹ → CO₂    solar.py
  → GeoJSON + summary
```

Three decisions worth knowing:

**Zoom 19, always.** `156543.0339 × cos(lat) / 2^19` is 0.30 m/px at the equator
and 0.27 at Bhopal. The model was trained on 0.3 m/px imagery. Serving at any
other zoom quietly changes the scale the network sees and costs accuracy.

**Area is geodesic, never Mercator.** Mercator overstates area by `1/cos²(lat)`
— +18% at Bhopal, +117% at Stockholm. Polygons are converted to lon/lat and
measured on the WGS84 ellipsoid.

**Preprocessing comes from the model's sidecar manifest**, not from a constant in
this code. `scripts/export_onnx.py` writes `<model>.json` next to the `.onnx`
with the normalisation mean/std, window and threshold. This is the permanent fix
for F-01, where training and inference normalised differently and every
prediction was wrong.

## Detection sensitivity — read this before changing it

The decision threshold is **not** a fixed number. It is chosen per region by
`coverage.THRESHOLD_BY_LEVEL`:

| region | threshold |
|---|---|
| inside a training city | 0.50 |
| near one (< 900 km) | 0.45 |
| outside | 0.40 |

Why: on Inria, 0.50 and 0.65 differ by 0.002 IoU — noise. Out of distribution
the model is under-confident, and raising the threshold to 0.60 measurably cost
**7–11% of detected roof area** across Bangalore and Bhopal. A threshold tuned
on in-distribution data does not transfer, so do not "optimise" it on Inria
again and ship the result globally. The UI discloses when the threshold was
chosen automatically, and the slider overrides it.

**TTA (the "High accuracy" toggle) is off by default** for the same reason:
worth +0.010 IoU on Inria, but −5% detected area in Bangalore, at 8× the cost.

## Solar exposure

Every result includes the location's **solar resource**, independent of the roof
or the system — the number that makes two places comparable. From PVGIS
`MRcalc`, averaged over the 2005–2020 climatology: monthly horizontal
irradiation (GHI), monthly optimal-plane irradiation, and monthly mean air
temperature.

| | annual GHI | peak sun hours | lowest month |
|---|---|---|---|
| Bangalore | 1934 kWh/m² | 5.3 h/day | July |
| Bhopal | 1872 kWh/m² | 5.1 h/day | **August (monsoon)** |
| Austin | 1815 kWh/m² | 5.0 h/day | December |
| Vienna | 1256 kWh/m² | 3.4 h/day | December |

Also available standalone at `GET /api/solar-resource?lat=..&lon=..`, so the
resource can be shown without running a detection.

## Honesty features

These are not decoration — they are the difference between a tool people can
trust and one that just looks confident.

* **Coverage banner.** Every result says whether the area is inside, near, or
  outside the model's training region, and says when that changed the
  threshold. Verified by eye: Austin (a training city) segments cleanly; Bhopal
  and Bangalore miss buildings. A user in India is told before they read the
  number.
* **"This is an estimate, not a site survey"** at the top of every result.
* **Per-roof confidence** — the model's mean predicted probability inside each
  polygon, not an invented score.
* **The full calculation chain** is shown, every step, with the value at each.
* **Every assumption is visible and adjustable**, with its source in the
  Model & method panel.
* **PVGIS failures are flagged**, not silently replaced by a guess.

## Configuration

| env var | default | what it does |
|---|---|---|
| `RSOLAR_TILE_PROVIDER` | `esri` | `esri` or `mapbox` |
| `RSOLAR_MAPBOX_TOKEN` | — | required for `mapbox` |
| `RSOLAR_MAX_TILES` | `256` | AOI size cap |
| `RSOLAR_CACHE` | `webapp/.cache` | PVGIS response cache |

**On imagery licensing.** Esri World Imagery is the default: global, no key,
works at z=19. It is appropriate for local and research use with attribution.
Before deploying this publicly, read your provider's current terms — several,
Google Maps Platform most strictly, prohibit running ML over their imagery or
caching derived products. Mapbox Raster Tiles has explicit programmatic-access
terms and a 200k tiles/month free tier.

## API

| route | |
|---|---|
| `POST /api/analyze` | → `202 {job_id}`; `threshold` null = auto by region, `tta` bool |
| `GET /api/solar-resource?lat=&lon=` | monthly GHI / tilt-plane / temperature |
| `GET /api/jobs/{id}` | job state, progress, result |
| `GET /api/model` | model card: architecture, metrics, limitations |
| `GET /api/assumptions` | defaults + the source of each |
| `GET /api/config` | client bootstrap |
| `GET /api/health` | |

Analysis is a job because a 1 km² AOI is ~180 tiles and tens of seconds; holding
an HTTP connection open for that is fragile and shows the user nothing.

```bash
curl -X POST localhost:8000/api/analyze -H 'Content-Type: application/json' \
  -d '{"bounds":{"west":-97.7450,"south":30.2680,"east":-97.7405,"north":30.2712}}'
curl localhost:8000/api/jobs/<id>
```

## Performance

Measured on this machine (Ryzen CPU, ONNX Runtime, no GPU): a 48-tile AOI —
2048×1536 px, 35 windows — takes **~11 s** end to end. Roughly 2 s of tile
fetching, 8 s of inference, 0.6 s of vectorisation.

## Deploying later

It is already a single container: `pip install -r requirements-webapp.txt`, copy
`webapp/`, run `uvicorn webapp.app:app`. Nothing needs torch — ONNX Runtime is
~50 MB against torch's ~2.5 GB. Before going public: check the tile ToS, put a
rate limit in front of `/api/analyze`, and move jobs to Redis if you want more
than one worker.
