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
  → threshold chosen for THIS area        calibration.py + regions.py
  → morphological open/close, kernel too  calibration.py -> geometry.py
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

The decision threshold is **not** a fixed number, and it is no longer a guess
either. Three sources of evidence, cheapest first, in `calibration.py`:

| step | cost | what it uses |
|---|---|---|
| 1. regional prior | free, offline | distance to a training city (`coverage.py`), clamped into the region's band (`regions.py`) |
| 2. histogram self-calibration | free | Otsu's valley in the model's own probability map |
| 3. reference calibration | ~1 min, once per ~5 km cell | real OpenStreetMap building outlines |

**Why a prior at all.** On Inria, 0.50 and 0.65 differ by 0.002 IoU — noise. Out
of distribution the model is under-confident, and raising the threshold to 0.60
cost **7–11% of detected roof area** across Bangalore and Bhopal. A threshold
tuned on in-distribution data does not transfer, so do not "optimise" it on
Inria again and ship the result globally.

**Why a histogram step.** A well-calibrated segmenter gives a bimodal
probability map — a mass near 0, a mass near 1 — and the right cut is the valley
between. Out of distribution the roof mode slides down and the valley slides
with it, so finding the valley tracks the model's own confidence collapse with
no labels anywhere. It is guarded hard: the map must genuinely be bimodal (Otsu
separability ≥ 0.88 — a *uniform* distribution already scores 0.75, so a lower
floor admits noise rather than weak evidence), the cut must imply a believable
1–75% roof coverage, and it may never move the prior by more than 0.12.

**Why OpenStreetMap, and the one asymmetry that makes it safe.** OSM is used for
**recall only, never precision**. It is badly incomplete in India, so "we
detected something OSM does not have" means nothing — but "OSM has a building
here" was drawn by a human and is almost always true, so a miss is real
evidence. Calibrating to *recover a known-real set* is immune to the
incompleteness that would wreck an IoU or precision target. Press **Calibrate to
this area**; the result is cached per ~5 km cell and every later analysis nearby
picks it up for free.

### What calibration found in Bangalore (2026-09-04)

The first real out-of-distribution recall measurement this project has. A dense
residential block in CV Raman Nagar, 142 OSM-mapped buildings, 35 tiles at z19:

| threshold | recall of mapped buildings |
|---|---|
| 0.28 (band floor) | 0.31 |
| 0.40 (the shipped prior) | 0.27 |
| 0.50 | 0.23 |
| 0.58 | 0.21 |

**The curve is almost flat, and that is the finding.** Dropping the threshold
from 0.50 all the way to 0.28 buys 8 points of recall. Broken down by footprint:

| roof size | n | recall | median probability |
|---|---|---|---|
| 0–50 m² | 24 | 0.08 | **0.006** |
| 50–100 m² | 47 | 0.17 | 0.049 |
| 100–200 m² | 41 | 0.37 | 0.171 |
| 200–500 m² | 13 | 0.46 | 0.343 |
| 500+ m² | 16 | 0.44 | 0.187 |

**56% of mapped buildings score below 0.10.** That is not under-confidence that
a lower cut can rescue — it is a confident negative. The model is *silent* on
small Indian rooftops, and no threshold anywhere reaches 0.006. This is why
`reference_calibrate` can return `verdict: "needs_finetuning"` and say so
instead of sliding the threshold down and manufacturing false positives out of
tree canopy. Fixing it needs training data, not a slider — see CLAUDE.md
"What to do next" #2.

Ruled out first, so nobody re-checks it: this is **not** an OSM-vs-Esri
georeferencing offset. Sweeping a rigid ±24 px (±7 m) shift of the footprints
moves recall between 0.17 and 0.29, peaking at 0.29 at +4.7 m — flat, and
within noise of the 0.27 at zero shift.

**TTA (the "High accuracy" toggle) is off by default** for the same reason the
threshold is regional: worth +0.010 IoU on Inria, but −5% detected area in
Bangalore, at 8× the cost.

## Regional defaults

`regions.py` answers *what is known about this place before we look at a single
pixel?* — currency, a representative tariff, installed cost per kWp, grid CO₂
intensity, the threshold band, and the built form to expect. The UI calls
`/api/region-profile` whenever the map settles and pre-fills the Assumptions
panel, so a Sydney user does not get quoted rupees. **A field the user has
edited is never overwritten.**

The split is deliberate: tabulate what is genuinely regional (money, grid), and
*measure* what is local (threshold, morphology). You cannot tabulate a threshold
per city — there are tens of thousands of them and the built form changes across
a single one.

Every economic figure is indicative and user-editable, and each carries an
`economics_confidence` of `medium` / `low` / `none` that the UI shows. An
unlisted location gets currency-neutral placeholders that admit they are
placeholders, which beats confidently wrong rupees.

Bounding boxes are rectangles and countries are not: India's box contains Sri
Lanka (carved back out), Nepal, Bhutan and most of Bangladesh (not — any
rectangle around them also swallows real Indian territory). The blast radius is
bounded, because those neighbours share India's threshold band and built form
anyway, so only the *economics labels* are wrong, and `matched` in the API
response lists every region that fired.

### Morphology is chosen per area too

The open/close kernel bridges gaps up to `k × metres_per_pixel`. The shipped
`k=3` bridges ~0.9 m at z19 — wider than the alley between two Indian row
houses, which is exactly the known defect where neighbours merge into one
polygon. `choose_morph_kernel` measures the median detected footprint and drops
to 2 px for small dense housing, rises to 4 px for warehouse roofs.

One subtlety: the measurement may only ever argue *downward*. It is biased
upward by the very merging the kernel is meant to reduce — on that Bangalore
block the median component read 119 m² against a regional expectation of 60,
because neighbours had already merged, and trusting it would have widened the
kernel and made the merging worse.

### Other reference sources

OSM is the default because it is global, free, key-less and human-drawn. For
bulk offline work, **Google Open Buildings** (CC-BY-4.0, ~1.8 B footprints
across the Global South including all of India) and **Microsoft Global ML
Building Footprints** are far more complete — but both are model-generated, so
they are fine for *calibration* and must never be used as an evaluation set.
Google Maps Platform is not an option: its terms prohibit running ML over its
imagery and caching derived products.

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
| `POST /api/analyze` | → `202 {job_id}`; `threshold` null = calibrated per area, `tta` bool |
| `POST /api/calibrate` | → `202 {job_id}`; measures the threshold here against OpenStreetMap |
| `POST /api/recalculate` | roof polygons + assumptions → the full estimate, no model involved |
| `GET /api/region-profile?lat=&lon=` | local currency/tariff/grid defaults, threshold band, any stored calibration |
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

## Correcting the detection

The model is the weakest link outside its training regions — it misses
buildings outright there, and no threshold recovers them
(`results/RESULTS.md`). Rather than hide that, the results panel has
**Correct the roofs**:

- **click** a roof to select it, **Delete** to remove a false positive
- **+ Add** then click corners, **Enter** (or double-click) to close a roof the
  model missed, **Esc** to cancel
- **Undo** / **Reset**, then **Recalculate from my edits**

`POST /api/recalculate` re-runs the whole solar chain — packing factor, PVGIS
yield, money, CO₂ — over the corrected polygons. Area is measured server-side
with `pyproj`, exactly as for detected roofs; the browser owns the geometry and
the server owns the measurement, so there is only ever one area implementation
(this project has been bitten by an area bug once already).

Edited results are labelled as such in the panel, user-drawn roofs are a
different colour and carry no fabricated confidence score, and the detection
settings shown still describe the original run.

## Which model is served

`webapp/models/` can hold several exported models. Selection is explicit, in
this order:

1. `RSOLAR_MODEL=<stem>` — pins one by file stem
2. the sidecar declaring `"default": true`
3. newest file (the last-resort guess)

Newest-wins alone was a booby trap: exporting a specialised checkpoint would
silently replace the general one for every user, and the symptom — every area
estimate changes — looks nothing like the cause.

```powershell
$env:RSOLAR_MODEL = "finetune_indian"   # the Indian cluster-envelope model
python -m webapp
```

**If you serve `finetune_indian`, drop the packing factor to ~0.5.** It
predicts building-cluster envelopes (alleys included), not footprints, so its
area runs ~2.1x the footprint model's on the same block. The sidecar carries
`recommended_packing_factor` and `/api/model` reports it.

## Pre-trained weights

The shipped model is published, so a fresh clone does not have to retrain:

**https://github.com/Parthesh10/rooftop-solar-potential-detection/releases/tag/v1.0-inria**

Download all three assets into `webapp/models/` (the `.json` sidecar is
required — it carries the normalisation constants, and the app refuses to load
a model without it):

```powershell
gh release download v1.0-inria --dir webapp/models
```

Or run `python scripts/export_onnx.py` if you have the `.pt` locally.
