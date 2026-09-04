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
labelme data/finetune_candidates/images --output data/finetune_candidates/images --nodata
```

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
