# Results

Two datasets, two sets of numbers. **Inria is the headline** — it is what the
shipped general model is trained and scored on. The Swiss DOP25 numbers below it
are the project's history and a secondary out-of-distribution check.

---

# THE SHIPPED MODEL — Inria, 2026-09-03 (re-measured 2026-09-04)

**Inria official val IoU 0.7712** at the shipped threshold of 0.50, rising to
**0.7809** with test-time augmentation. Target was ≥ 0.72. ✅

> The 0.7233 quoted below in "The Inria run" is the *training-time* number, a
> mean of per-window IoUs. It is the same model on the same split; pooled IoU is
> the comparable-to-literature figure. See "Post-hoc tuning" for why they differ.

| | |
|---|---|
| Weights | `results/unetpp_effb0_inria_20260903.pt` (26 MB) |
| Architecture | **U-Net++ / EfficientNet-B0**, ImageNet-pretrained encoder |
| Parameters | 6.6 M |
| Trained on | Inria official split — 155 tiles (austin, chicago, kitsap, tyrol-w, vienna), tiles 6–36 per city |
| Scored on | Inria official val — 25 tiles, tiles 1–5 per city, **never seen in training** |
| Input | 512×512 @ 0.3 m/px, ImageNet normalisation |
| Labels | **building footprints** — roof extent, not installable area |

| metric | shipped (thr 0.50) | with 8x TTA (thr 0.60) |
|---|---|---|
| **IoU** (pooled) | **0.7712** | **0.7809** |
| F1 | 0.8708 | 0.8770 |
| precision | 0.8454 | 0.8648 |
| recall | 0.8978 | 0.8895 |

Config: `--window 512 --samples-per-tile 48 --pos-weight 2.4 --dice-weight 0.6
--epochs 60 --batch-size 16 --lr 3e-4 --patience 12`, AdamW + cosine with
5-epoch warmup, D4 + photometric augmentation. Kaggle T4, fp16, ~4.8 h/config.

## The Inria run

| run | arch / encoder | best epoch | **val IoU** | F1 | P | R |
|---|---|---|---|---|---|---|
| **I2** | **U-Net++ / efficientnet-b0** | 49 | **0.7233** | 0.818 | 0.823 | 0.857 |
| I1 | U-Net / resnet34 | 59 | 0.7178 | 0.813 | 0.824 | 0.850 |

*(These are the per-window means the training loop printed. Pooled, I2 is
0.7712 — see "Post-hoc tuning" below. I1 was not re-measured pooled.)*

Both cleared the target; effb0 wins by 0.6 points at **a quarter the parameters**
(6.6 M vs 24.4 M), which is why it ships — smaller model, faster inference,
better score.

### What it showed

* **Data was the constraint, exactly as diagnosed.** The same architecture family
  went from ~0.52–0.57 on 420 Swiss tiles to **0.72** on Inria. Nothing about the
  loss, the schedule or the augmentation changed materially.
* **The pretrained encoder pays off once there is data to feed it.** On Swiss it
  tied the scratch net (see below); on Inria it is the whole result.
* **Healthy generalisation gap.** Train IoU 0.838 vs val 0.723 — a real ~0.11
  gap, not the 0.13+ overfit the Swiss runs showed on a 58-tile val set.
* **Converged cleanly.** Val IoU plateaued at ~0.722 from epoch ~42 and the
  cosine schedule annealed to zero without divergence.

Published Inria building-segmentation work sits ~0.78–0.82 with much larger
models and multi-scale inference. 0.723 from a 6.6 M-parameter model in one
4.8 h run is an honest, defensible result.

### Post-hoc tuning, 2026-09-04 — and a metric correction

**The training number was understated.** Training reported val IoU 0.7233, which
is a **mean of per-window IoUs**: it weights a 512x512 window holding one small
roof the same as a dense city block, so sparse windows dominate it. Pooled
(global) intersection-over-union on the same model and the same split is
**0.7712** — and pooled is the metric the Inria benchmark itself reports, so it
is the one comparable to published work. Both numbers are in the manifest.

A threshold and TTA sweep then bought real gains for no retraining
(`scripts/eval_inria.py`, 25 val tiles / 2025 windows):

| setting | IoU | F1 | precision | recall |
|---|---|---|---|---|
| threshold 0.50, no TTA | 0.7712 | 0.8708 | 0.8454 | 0.8978 |
| threshold 0.65, no TTA | 0.7733 | 0.8722 | 0.8639 | 0.8806 |
| **threshold 0.60 + 8x dihedral TTA** | **0.7809** | **0.8770** | **0.8648** | **0.8895** |

### The threshold does not transfer out of distribution

Raising the shipped default from 0.50 to 0.65 on the strength of the table above
was **wrong**, and measurably so. On Indian cities the model is under-confident,
so a stricter cut deletes real buildings:

| area | roof m² @ 0.50 | @ 0.60 | change |
|---|---|---|---|
| Bangalore, Jayanagar | 73,697 | 68,658 | **-7%** |
| Bangalore, Indiranagar | 55,923 | 49,569 | **-11%** |
| Bangalore, Manyata | 50,084 | 47,179 | -6% |
| Bhopal, MANIT | 3,872 | 3,569 | -8% |

Since 0.50 and 0.65 differ by only 0.002 IoU in-distribution — noise — but by
7-11% of recall out of distribution, **0.50 is the correct global default** and
the app now picks per region (`webapp/coverage.THRESHOLD_BY_LEVEL`): 0.50 inside
the training cities, 0.45 near them, 0.40 outside. Users are told when this
happens and can override it.

**TTA does not transfer either.** On Inria it is worth +0.010 IoU; on Bangalore
it measured +0.4% area at threshold 0.30 and **-5%** at 0.50, for 8x the compute.
It ships opt-in, not on by default.

Two things this ruled out along the way, both by measurement rather than
argument: there is **no georeferencing offset** (mask vs re-projected polygons
round-trips at IoU 0.92 with best-fit shift exactly (0,0)), and the model does
**not** have a bright-roof blind spot (pixels above luminance 200 are detected at
34% versus 23% for mid-tones). The gap is genuine domain shift — Bangalore
imagery has contrast std 78 against Inria's 40-52.

### Known limitations — read before trusting a number

* **Five Western cities only**: Austin, Chicago, Kitsap County WA, Vienna,
  Tyrol. US suburban + Alpine European rooftops. Performance on dense low-rise
  Indian, African or East-Asian rooftops is **untested**.
* **Footprint, not installable area.** The model outputs roof extent. Usable PV
  area needs a packing factor (0.70–0.80 default) for setbacks, walkways,
  parapets, tanks and inter-row spacing. The web app exposes this as a slider.
* **0.3 m/px.** Serve at Web Mercator z=19 (≈0.30 m/px at the equator, 0.27 at
  23°N) to keep train and deploy resolution matched.

---

# Swiss DOP25 — project history and secondary eval

All numbers in this section use the **leakage-free geographic split** (420 train
/ 58 val / 74 test, zero tiles adjacent across splits) and the **corrected
metric harness** (`model.eval()` on, no empty-tile IoU inversion). Anything
measured any other way is not comparable — see "Why the 2023 checkpoints are not
a baseline" below.

> **Why this dataset stopped being the target (2026-08-29):** 420 training tiles
> proved too small to benefit from a pretrained encoder — the encoder sweep below
> is the evidence. The project moved to Inria for the general model.

## Encoder sweep — 2026-08-29 (Kaggle T4, fp16, ~45 min)

Four architectures, loss recipe fixed at `pos_weight 2.4 / dice_weight 0.7`,
80 epochs each. **Question: does an ImageNet encoder beat the scratch U-Net?**
**Answer on Swiss: no.**

| run | arch / encoder | val IoU | **test IoU** | P | R |
|---|---|---|---|---|---|
| U1 | U-Net / **resnet34** (ImageNet) | 0.6927 | **0.5653** | 0.727 | 0.742 |
| U0 | U-Net / scratch *(control)* | 0.5980 | 0.5651 | 0.676 | 0.779 |
| U2 | U-Net++ / efficientnet-b0 | 0.6680 | 0.5191 | 0.732 | 0.665 |
| U3 | DeepLabV3+ / efficientnet-b2 | 0.6420 | 0.5087 | 0.715 | 0.665 |

### What it showed

* **resnet34 and scratch tie on the held-out test set** (0.5653 vs 0.5651).
  The two heavier pretrained models (effb0, effb2) score *worse* than scratch.
* **The pretrained models overfit the val block.** resnet34's val→test drop is
  **13 points** (0.69 → 0.57); the scratch net's is 4 (0.60 → 0.57). A 58-tile
  val set from one geographic block is not enough to select on — the pretrained
  encoder fits it and does not carry to the test block.
* **The metric has ≈ ±2 IoU of noise.** The scratch control here scored 0.5651;
  the *same config* on a P100 in fp32 (sweep D, 2026-08-22) scored 0.5442. The
  hardware/precision change alone moved it ~2 points, so 0.5653-vs-0.5442 is
  **not** a real gain.
* **Takeaway:** 420 tiles is the binding constraint, exactly as suspected —
  but the fix is *more data*, not a better encoder on the same data. On to Inria.

Weights kept for reference: `unet_swiss_resnet34_20260829.pt` (U1). Not promoted
over `unet_swiss_geo_D_pw2.4_d0.7.pt` — they are tied within noise.

## Swiss baseline — sweep D, 2026-08-22

**Test IoU 0.5442.** Weights: `unet_swiss_geo_D_pw2.4_d0.7.pt`
(run `D_pw2.4_d0.7`, best epoch 69 of 80).

| split | IoU | F1 | acc | precision | recall |
|---|---|---|---|---|---|
| val | 0.5948 | — | — | — | — |
| **test** | **0.5442** | — | — | 0.655 | 0.777 |

Config: `--pos-weight 2.4 --dice-weight 0.7 --epochs 80 --batch-size 16
--lr 3e-4 --patience 40`, AdamW + cosine with 5-epoch warmup, D4 + photometric
augmentation, 224x224 random crops.

## The sweep that produced it

Four configs, Kaggle Tesla P100, ~40 min total. All ran the full 80 epochs.

| run | pos_weight | dice | best epoch | val IoU | **test IoU** | P | R |
|---|---|---|---|---|---|---|---|
| **D** | 2.4 | 0.7 | 69 | 0.5948 | **0.5442** | 0.655 | 0.777 |
| B | 2.4 | 0.5 | 58 | 0.6001 | 0.5369 | 0.673 | 0.751 |
| C | 1.0 | 0.5 | 49 | 0.5924 | 0.5003 | 0.706 | 0.652 |
| A | auto (~5.9) | 0.5 | 46 | 0.5693 | 0.4470 | 0.650 | 0.642 |

### What it showed

**`pos_weight` was the dominant factor, worth ~9 IoU points.** The auto-estimate
(negatives/positives ≈ 5.9) over-predicts badly. Its square root, 2.4, is much
better; dropping weighting entirely (1.0) gives the best precision of the four
(0.706) but loses so much recall that IoU falls again.

**An earlier hypothesis was wrong.** The previous run (test IoU 0.4656)
early-stopped at epoch 27 with its best at 12, and I attributed the weak result
to under-training. Run A tests that directly: same `pos_weight`, but 80 full
epochs. It scored **0.4470 — worse than the early-stopped run.** Training
longer with a bad loss weight actively hurt. The fix was the weight, not the
duration.

## History

| date | model | dataset | IoU | notes |
|---|---|---|---|---|
| 2026-09-04 | **U-Net++ / effb0** | **Inria val (pooled)** | **0.7712** | **shipped model** — 0.7809 with TTA |
| 2026-09-03 | U-Net++ / effb0 | Inria val (per-window) | 0.7233 | same model, weaker averaging |
| 2026-09-03 | U-Net / resnet34 | Inria val | 0.7178 | runner-up, 3.7× the params |
| 2026-08-29 | U-Net / resnet34 | Swiss test | 0.5653 | encoder sweep; ties scratch |
| 2026-08-29 | U-Net / scratch (T4/fp16) | Swiss test | 0.5651 | same config as sweep D, +2 from hardware/noise |
| 2026-08-22 | sweep D (P100/fp32) | Swiss test | 0.5442 | Swiss baseline |
| 2026-08-22 | first clean run | Swiss test | 0.4656 | early-stopped at 27, best @ 12 |
| 2026-08-11 | 2023 `path raise 130.pt` | Swiss test | (0.4946) | **contaminated — not a baseline** |

Inria and Swiss IoU are **not comparable**: different label semantics (footprint
vs available-roof-area), different resolution (0.30 vs 0.25 m/px), different
countries. The jump from 0.54 to 0.72 is mostly "100× more training data", not a
like-for-like improvement.

## Why the 2023 checkpoints are not a baseline

`model/path raise *.pt` were trained on a *random* split of the same 574 tiles.
The tiles are 62.5 m apart, so a random split leaks by construction (F-05) — and
worse, any freshly drawn evaluation set is mostly tiles they already trained on:

    new val  (58 tiles): 41 were in the 2023 training set -> 71%
    new test (74 tiles): 51 were in the 2023 training set -> 69%

So all three of their numbers are contaminated to an unknown degree:

* **0.5566** — reported in the 2023 project report. Also measured with
  BatchNorm in *training* mode (F-03) and with the empty-tile IoU inversion
  (F-08), so it is not a valid measurement even ignoring the leak.
* **0.5170** — the same checkpoint re-scored with the fixed harness.
* **0.4946** — the same checkpoint on the clean geographic split. Looks
  competitive with 0.5442 but it has seen 69% of that test set.

There is no way to re-partition this dataset that gives those checkpoints an
honest test set.

## Reproducing

```powershell
git push origin main
.\.venv\Scripts\python.exe -m kaggle kernels push -p kaggle --accelerator gpuT4x2
.\.venv\Scripts\python.exe -m kaggle kernels output partheshgupta/rooftop-solar-u-net-training -p kaggle_out
```

Locally (~2.5 h, batch 4 to stay inside 4 GB):

```powershell
.\.venv\Scripts\python.exe scripts\train_swiss.py --pos-weight 2.4 --dice-weight 0.7 --patience 40
```

## Next

1. **Broaden geography** — Inria is 5 Western cities. Google Open Buildings
   (Global South, CC-BY) and SpaceNet (Rio / Shanghai / Khartoum) add the
   diversity a genuinely global model needs. Hand-label a small eval set per
   new region; never evaluate on auto-generated labels.
2. **Threshold + TTA** — cheap post-hoc gains, no retraining. `evaluate.py
   --tta` averages the 8 dihedral transforms (+1–2 IoU, 8× cost).
3. **Multi-class ARA** — reaching true *available* rooftop area (excluding
   chimneys, skylights, existing panels) needs multi-class labels. Research item.

---

## Out-of-distribution recall, measured (2026-09-04)

Every number above is Inria. This is the first measurement of the shipped model
on rooftops it was never trained on, against **human-drawn** reference
footprints rather than auto-generated ones.

Method: `webapp/calibration.py`, `POST /api/calibrate`. A dense residential
block in CV Raman Nagar, Bangalore (77.654–77.658 E, 12.984–12.987 N), 35 tiles
at z19, 142 OpenStreetMap building ways, 141 of them wholly inside the mosaic. A
footprint counts as recovered when the median model probability inside it clears
the threshold.

| threshold | recall of mapped buildings | positive pixel fraction |
|---|---|---|
| 0.28 | 0.31 | 0.137 |
| 0.40 | 0.27 | 0.124 |
| 0.50 | 0.23 | 0.110 |
| 0.58 | 0.21 | 0.101 |

By footprint size, at threshold 0.40:

| roof size | n | recall | median probability inside |
|---|---|---|---|
| 0–50 m² | 24 | 0.08 | 0.006 |
| 50–100 m² | 47 | 0.17 | 0.049 |
| 100–200 m² | 41 | 0.37 | 0.171 |
| 200–500 m² | 13 | 0.46 | 0.343 |
| 500+ m² | 16 | 0.44 | 0.187 |

**56% of mapped buildings have a median probability below 0.10.** The failure is
not a badly placed cut point — the model returns a confident negative on small
Indian rooftops. Recall moves 8 points across the entire usable threshold range,
so thresholding is not the lever. Detected roof area rose 14,125 → 17,927 m²
(+27%) going from 0.50 to 0.28, while the building *count* fell 30 → 27: the
lower cut merges neighbours rather than finding new houses.

**This is a recall figure against an incomplete reference, not an IoU and not a
precision figure.** OSM under-maps Indian residential blocks, so a detection OSM
lacks is not evidence of a false positive. The asymmetry is the point: an OSM
building was drawn by a human, so a miss is real.

Not a georeferencing artefact: sweeping a rigid ±24 px (±7 m) shift of the OSM
footprints moves recall between 0.17 and 0.29, peaking at 0.29 at +4.7 m against
0.27 at zero shift — flat, and within noise. Consistent with the 2026-09-03
finding of no mask-to-polygon offset.

**Do not compare 0.27 to 0.7712.** One is recall against human-mapped footprints
in Bangalore; the other is pooled IoU on Inria val. Different metric, different
labels, different continent.

---

## Fine-tuning on hand-labelled Indian tiles (2026-09-05)

**The hypothesis in the previous section was right: it was a data problem, not
a capacity one.** 80 hand-labelled tiles, ~4 minutes of training on a GTX 1650,
and the Bangalore failure inverts.

Recipe: `scripts/finetune_indian.py` — the shipped Inria checkpoint, encoder
frozen for 5 epochs then unfrozen, lr 1e-5, pos_weight 2.4, 256px random crops,
batch 4. Early-stopped at epoch 17 (best epoch 9), peak VRAM 1.3 GB.
Held-out val IoU on 12 unseen Indian tiles: **0.5143**.

Measured against the *same* 141 OpenStreetMap footprints in CV Raman Nagar used
for the baseline above, at threshold 0.40:

| roof size | n | recall before | recall after |
|---|---|---|---|
| 0–50 m² | 24 | 0.08 | **0.83** |
| 50–100 m² | 47 | 0.17 | **0.81** |
| 100–200 m² | 41 | 0.37 | **0.98** |
| 200–500 m² | 13 | 0.46 | **0.92** |
| 500+ m² | 16 | 0.44 | **0.88** |
| **overall** | 141 | **0.27** | **0.88** |
| **silent (<0.10)** | | **56%** | **4.3%** |

Both columns are measured at the **same fixed threshold 0.40** — an earlier
draft of this table compared each model at its own calibrated threshold, which
flattered both sides. At 0.50 the picture is the same: overall 0.24 → 0.84.

The calibration verdict moves from `needs_finetuning` to `calibrated`. The
0–50 m² band — the confident-negative case that no threshold could reach —
went from a median probability of **0.006 to 0.869**.

### The catch: these labels are cluster envelopes, not footprints

The labeller merged adjacent buildings into single polygons and did not cut out
the alleys between them, to get through 93 tiles in reasonable time. Measured
over the 835 drawn polygons:

| | value |
|---|---|
| median polygon | 381 m² ≈ 6.4 houses |
| mean polygon | 714 m² ≈ 12 houses |
| polygons implying >20 houses | 14% |
| largest | 15,111 m² (a whole colony block) |

So the fine-tuned model predicts **built-up cluster envelopes**, and its
predicted area on the baseline block is **+115%** (23,656 → 50,742 m²). That is
not an error — it is what it was taught — but it has two hard consequences:

1. **The packing factor must absorb it.** `model/manifest.json` records
   `recommended_packing_factor: 0.5` for this checkpoint against 0.70–0.75 for
   the footprint model. Serving it without that change roughly doubles every
   kWh and money figure.
2. **Its building *count* is meaningless**, and its Inria IoU is not comparable
   with 0.7712.

### Inria regression check

Same 5 Inria val tiles, threshold 0.50:

| model | IoU | F1 | precision | recall |
|---|---|---|---|---|
| shipped (footprints) | **0.7996** | 0.8886 | 0.8756 | 0.9020 |
| fine-tuned (envelopes) | 0.6540 | 0.7908 | 0.7010 | 0.9072 |

**Recall is unchanged (0.902 → 0.907); precision falls (0.876 → 0.701).** That
is the exact signature of an envelope model scored against footprint labels —
it covers the buildings and then some. It is *not* catastrophic forgetting: the
model did not stop finding buildings, it started outlining them differently.

Because of that regression on Western imagery, the fine-tuned checkpoint is
**not** the default. `webapp/models/` now holds both, the sidecar marked
`"default": true` wins regardless of file age, and
`RSOLAR_MODEL=finetune_indian` serves the Indian one.

### What would settle it

The open question is whether a model can be good at both. Joint training on
Inria footprints plus Indian tiles is the obvious next run, but the two label
sets disagree about what a "roof" is, so mixing them teaches a contradiction.
Either relabel a subset of the Indian tiles per-building, or train a
two-headed model. That design question is unresolved and worth more than a
speculative GPU run.

---

## Relabelling from OpenStreetMap, and the three-way comparison (2026-09-05)

The envelope labels worked but predicted the wrong quantity. The obvious fix —
keep the good tiles, drop the merged ones — **does not work**, and the
measurement is unambiguous:

| separator | merged polygons | single-building polygons |
|---|---|---|
| solidity | 1.00 | 1.00 |
| rectangularity | 0.89 | 0.89 |
| area (median) | 859 m² | 359 m² |

Shape carries essentially no signal. The best area cutoff (490 m²) keeps 63% of
true single-building polygons while also keeping 14% of merges, and requiring a
tile to be mostly footprint-like leaves **11–20 usable tiles out of 91**. The
merging is pervasive; there is no clean subset hiding inside it.

Ground truth for that table came from OpenStreetMap: 4,828 building footprints
across the four AOIs, counting how many distinct OSM buildings fall inside each
hand-drawn polygon (≥2 = a definite merge). 197 polygons were definite merges,
126 contained exactly one building, 512 contained none.

### OSM is not a drop-in replacement either

| | share of tile area |
|---|---|
| hand labels (envelopes) | 31.8% |
| OSM buildings | 16.2% |
| overlap | 8.5% |

OSM covers only **26.7%** of what the labeller marked built-up, and **47.6% of
OSM buildings fall outside any hand polygon** — the labeller covered part of
each tile, not all of it. Training on OSM as-is would call a great deal of real
roof "background", which is precisely the failure being fixed.

### Using each source for what it is reliable at

`scripts/build_osm_labels.py`:

| label | source | why |
|---|---|---|
| **positive** | inside an OSM building | human-drawn, one polygon per building |
| **ignore** | hand envelope minus OSM | genuinely unknown — an alley, or a roof nobody mapped |
| **negative** | everything else | |

103 tiles (more than the 80 hand-labelled ones, because OSM covers tiles the
labeller skipped), 23% positive, 17% ignored. Validation holds out the whole
`bangalore_cvraman_a` AOI, which contains the CV Raman Nagar benchmark block —
so the numbers below are measured on data the model never saw.

### The comparison that decides which model to ship

Pooled IoU against the 142 OSM footprints in the held-out block, threshold 0.50.
OSM's reference area there is 30,436 m².

| model | IoU | precision | recall | predicted area | ÷ OSM |
|---|---|---|---|---|---|
| shipped (Inria) | 0.254 | 0.491 | 0.345 | 21,373 m² | 0.70× |
| envelope fine-tune | **0.444** | 0.501 | 0.796 | 48,349 m² | 1.59× |
| **OSM fine-tune** | 0.409 | 0.502 | 0.689 | 41,825 m² | **1.37×** |

**Precision is ~0.49 for all three.** That is not a coincidence and it is not
model quality — it is OpenStreetMap's incompleteness. Roughly half of what
*any* model predicts is simply not mapped, so precision here measures the
reference, not the prediction. Read the IoU and recall columns; treat precision
as a constant.

The envelope model wins on raw IoU, but it buys that with area: a blob painted
over a cluster covers every OSM building inside it for free. The OSM model
gives up 0.035 IoU for a 14-point reduction in area inflation **and** correct
footprint semantics, so the ordinary packing factor applies instead of 0.5.
That is the better trade for an app that multiplies area into money.

### Neither is a safe global default

| model | Inria IoU (5 val tiles, thr 0.50) |
|---|---|
| shipped | **0.7996** |
| envelope fine-tune | 0.6540 |
| OSM fine-tune | 0.6394 |

Fine-tuning on ~100 narrow tiles costs **~0.16 Inria IoU regardless of label
source**. Better labels fixed *what* the model predicts; they did nothing about
forgetting. Both fine-tunes stay opt-in behind `RSOLAR_MODEL`.

That is what `kaggle_joint/` is for, and it is only coherent now that both
halves mean the same thing: Inria footprints plus the OSM-relabelled Indian
tiles, in one training run, with the Indian tiles repeated 20x so 103 of them
are ~22% of an epoch rather than 1.4%. Launched 2026-09-05.
