# Fine-tuning on hand-labelled Indian rooftops

CLAUDE.md's "cheapest first experiment" (next-steps #2), now with the tooling
to run it. Context: `results/RESULTS.md`'s "Out-of-distribution recall,
measured" section found that the shipped model recovers only 27% of mapped
buildings in a dense Bangalore block, and 56% of them score below probability
0.10 — a confident negative, not an under-confident one. No amount of
threshold tuning fixes a confident negative; only more representative training
data does. This is that data-collection and fine-tuning pipeline.

## The three steps

```
scripts/select_finetune_tiles.py   -->  labelme (external tool)  -->  process_data/labelme_to_masks.py  -->  scripts/finetune_indian.py
   picks which tiles to label            you draw the polygons        turns polygons into PNG masks         fine-tunes the shipped checkpoint
```

### 1. Select candidate tiles

```powershell
python scripts/select_finetune_tiles.py --out data/finetune_candidates
```

This does **not** pick tiles at random. It fetches real imagery for a few
named AOIs (currently four boxes across Bangalore's CV Raman Nagar and central
Bhopal — the same two cities `webapp/coverage.py` already treats as
reference points), runs the shipped model, and scores every 512x512 window
two ways:

- **built-up-ness** (grayscale std) — drops blank fields, water, uniform tree
  canopy. Not worth a human's time.
- **model uncertainty** — how far the window's probabilities sit from a
  confident 0 or 1. Biased toward the windows most likely to hold the
  buildings the model is silently missing, because that is exactly the failure
  `results/RESULTS.md` measured. A minority of picks are still the model's
  *confident* windows, so the labelled set is not purely adversarial.

Output: `data/finetune_candidates/images/*.png` (120 by default — label about
100 of them; some will turn out low-quality) and a `manifest.csv` recording
each tile's real-world bounds and scores.

Add AOIs by editing `DEFAULT_AOIS` in the script — every box must be sized so
`(west,south,east,north)` stays under the 240-tile cap. The script prints the
tile count for anything you add; if it errors, shrink the box.

### 2. Hand-label with labelme

```powershell
pip install labelme
labelme data/finetune_candidates/images --output data/finetune_candidates/images
```

(Older labelme took a `--nodata` flag to keep the image out of the JSON. Newer
versions removed it — that is now the default, and `--with-image-data` is the
opt-in. `labelme_to_masks.py` handles either.)

Draw one polygon per rooftop, label it `building` (labelme's default single
class is fine — it doesn't matter what you type as long as it isn't one of
`not-building` / `ignore` / `exclude`, which `labelme_to_masks.py` treats as
"draw this polygon but don't count it", for flagging something you don't want
counted without deleting it). `Ctrl+S` writes `<stem>.json` next to the image;
`labelme`'s own hotkeys (`Ctrl+N` for a new polygon, `A`/`D` to move between
images) make ~100 tiles a few hours of work, not days — this is genuinely
manual and there is no shortcut that doesn't reintroduce the auto-generated-
label problem CLAUDE.md already warns against.

You do not have to label every candidate. Skip anything ambiguous, cloudy, or
outside your judgement — `labelme_to_masks.py` only converts tiles that
actually got a `.json` with at least one polygon.

### 3. Convert labels to masks

```powershell
python -m process_data.labelme_to_masks `
    --images data/finetune_candidates/images `
    --annotations data/finetune_candidates/images `
    --out data/finetune_indian
```

Writes `data/finetune_indian/images/<stem>.png` +
`data/finetune_indian/labels/<stem>_label.png` — the exact layout
`process_data.data_loader.DataLoaderSegmentation` already expects (same
convention as the Swiss DOP25 set). Prints how many tiles converted vs were
skipped (no polygons, or an error); fewer than 20 and `finetune_indian.py`
will refuse to run.

### 4. Fine-tune

```powershell
python scripts/finetune_indian.py
```

Loads `results/unetpp_effb0_inria_20260903.pt`, freezes the encoder for the
first 5 epochs (`--freeze-epochs`), then unfreezes and continues to 25 total
(`--epochs`) at `lr=1e-5`. The encoder starts frozen because ~100 tiles is
nowhere near enough to retrain a 5.3M-parameter EfficientNet-B0 encoder from
scratch without either overfitting or destroying the Inria-trained features
being fine-tuned *from* — the decoder adapts first, then the encoder is
allowed to drift once the decoder has somewhere sensible to send gradients.

`--pos-weight` defaults to **2.4**, not an auto-estimate from the fine-tuning
set. CLAUDE.md hard-won fact #3: an auto-estimated pos_weight (~5.9 on the
Swiss set) over-predicted and cost 9 IoU points; 2.4 is the value already
proven on this project, and a ~100-tile set is too small to re-derive a better
number from scratch.

Output: `results/finetune_indian.pt` plus a `.metadata.json` recording the
exact recipe (base checkpoint, epochs, lr, val IoU on the held-out Indian
tiles). Pause / stop / resume works exactly as in `train_swiss.py` — see
`runs/finetune_indian/CONTROL.md`.

## After fine-tuning: did it work, and did it forget Inria?

Two checks, in order:

```powershell
# 1. Catastrophic forgetting check — must stay close to 0.7712
python scripts/eval_inria.py --limit-tiles 5

# 2. Export and re-measure against the SAME Bangalore block used for the
#    baseline in results/RESULTS.md, so the before/after is apples-to-apples
python scripts/export_onnx.py --ckpt results/finetune_indian.pt
python -m webapp   # then POST /api/calibrate on 77.654-77.658E, 12.984-12.987N
```

Compare the new `recall_by_size` breakdown against the baseline table in
`results/RESULTS.md`. The number that matters most is the 0-50 m² recall (0.08
at baseline) and the fraction of buildings scoring below 0.10 (0.56 at
baseline) — those are the confident negatives no threshold could ever fix, and
they're the only numbers this experiment can actually move.

## What this does and does not prove

~100-150 tiles across two cities cannot make the model generalise to India.
What it tests is narrower and still useful: **is the Bangalore failure fixable
with more representative data, or is it something deeper** (architecture,
resolution, imagery source)? If fine-tuning meaningfully lifts the 0-50 m²
recall without cratering the Inria score, that is the strongest evidence yet
for CLAUDE.md's next-steps #2 (Google Open Buildings / SpaceNet at scale,
CC-BY, ~1.8B footprints across the Global South) being worth the larger
investment. If it doesn't move much, that argues architecture or resolution is
the real constraint, not sample count — worth knowing before committing to a
much bigger labelling effort.

---

## What actually happened (2026-09-05)

Run once, for real: 93 tiles labelled, 80 with usable polygons, ~4 minutes of
training on a GTX 1650. Against the same 141 OpenStreetMap footprints as the
baseline, both models at a fixed threshold of 0.40:

| roof size | n | recall before | recall after |
|---|---|---|---|
| 0–50 m² | 24 | 0.08 | **0.83** |
| 50–100 m² | 47 | 0.17 | **0.81** |
| 100–200 m² | 41 | 0.37 | **0.98** |
| 200–500 m² | 13 | 0.46 | **0.92** |
| 500+ m² | 16 | 0.44 | **0.88** |
| **overall** | 141 | **0.27** | **0.88** |
| **silent (<0.10)** | | **56%** | **4.3%** |

Verdict: `needs_finetuning` → `calibrated`. The question this experiment was
built to answer — data problem or capacity problem — is answered: **data.**

### Read this before trusting the numbers

The labelling merged adjacent buildings and kept the alleys between them, so
the model learned **cluster envelopes, not footprints**:

- predicted area is **+115%** vs the footprint model on the same block
- so `packing_factor` must drop to **~0.5** (recorded as
  `recommended_packing_factor` in `model/manifest.json`), or every kWh and
  money figure roughly doubles
- the building **count** it reports is meaningless
- its Inria IoU (0.654 on 5 tiles, vs the shipped 0.800) is **not** a
  regression in detection — recall is unchanged and precision falls, which is
  what an envelope model does against footprint labels

It is therefore **not the default model**. Serve it deliberately:

```powershell
$env:RSOLAR_MODEL = "finetune_indian"
python -m webapp
```

### If you label another batch

Draw **one polygon per building** and cut out the alleys, if you want a model
whose area is directly usable. It is slower per tile, and the payoff is that
the packing factor goes back to meaning what it means everywhere else. The
current 80-tile set proves the approach works; a footprint-labelled set of the
same size would make it *shippable*.
