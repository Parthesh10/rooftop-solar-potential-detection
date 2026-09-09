# Results

Two datasets, two sets of numbers. **Inria is the headline** — it is what the
shipped general model is trained and scored on. The Swiss DOP25 numbers below it
are the project's history and a secondary out-of-distribution check.

---

# THE INRIA-ONLY MODEL — 2026-09-03 (re-measured 2026-09-04)

> **This was the shipped model from 2026-09-03 to 2026-09-07.** The default is
> now `joint_v3_world25_20260908.pt` (25 areas on four continents) — see "Round
> 3" and "The first real IoU outside Inria" at the end of this file. Everything
> in this section still describes `unetpp_effb0_inria_20260903.pt`, and
> **0.7712 is this model's number, not the current default's** (which scores
> 0.7671). Quote it as such.
>
> For scale, on human-drawn rooftop labels in rural Karnataka this model scores
> **IoU 0.150** against the current default's **0.579**, and recovers 17% of the
> actual roof area against the default's 102%.

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

| model | Inria IoU (5 val tiles, thr 0.50) | full 25-tile val (2026-09-06) |
|---|---|---|
| shipped | **0.7996** | **0.7712** |
| envelope fine-tune | 0.6540 | 0.6828 |
| OSM fine-tune | 0.6394 | 0.6708 |

Fine-tuning on ~100 narrow tiles costs Inria IoU regardless of label source.
Better labels fixed *what* the model predicts; they did nothing about
forgetting. Both fine-tunes stay opt-in behind `RSOLAR_MODEL`.

> **The magnitude here was overstated and is corrected below.** The 5-tile
> column made the cost look like ~0.16. Measured on the full 25-tile official
> val split it is **~0.10**. The right-hand column was added on 2026-09-06; use
> it, and see "A correction to an earlier number" at the end of this file.

That is what `kaggle_joint/` is for, and it is only coherent now that both
halves mean the same thing: Inria footprints plus the OSM-relabelled Indian
tiles, in one training run, with the Indian tiles repeated 20x so 103 of them
are ~22% of an epoch rather than 1.4%. Launched 2026-09-05.

---

## Collecting the three Kaggle runs, and why the default did not change (2026-09-06)

Three jobs were launched on 2026-09-05. Two finished, one never could.

| kernel | what it tested | outcome |
|---|---|---|
| `kaggle_joint/` | Inria + all 103 OSM-relabelled Indian tiles, repeated 20x | complete, 46 epochs |
| `kaggle_inria_b0/` | the same model as shipped but `samples-per-tile` 96 instead of 48 | complete, 50 epochs |
| `kaggle_inria_b3/` | EfficientNet-B3, `samples-per-tile` 28 | cancelled in epoch 0 |

### Every model, measured the same two ways

Comparing these needed a second harness. `scripts/eval_inria.py` says whether a
model still works where it was trained; it says nothing about the place this
project actually wants to serve. `scripts/eval_india_osm.py` is new and supplies
the other half — one mosaic of the CV Raman Nagar block fetched **once** and
reused for every model, so a difference between two rows is the model and not
the imagery.

It reproduces the 2026-09-05 ad-hoc figures exactly (shipped 0.254 / 0.491 /
0.345 / 21,373 m²; OSM fine-tune 0.409 / 0.502 / 0.689 / 41,825 m²; joint v1
pixel recall 0.460), which is the reason to trust the new rows below.

| model | Inria pooled IoU @0.50 | Bangalore footprint recall | silent (<0.10) | area ÷ OSM |
|---|---|---|---|---|
| **shipped** (spt 48) | 0.7712 | 0.241 | 0.560 | 0.70× |
| B0 control (spt 96) | **0.7843** | 0.191 | 0.716 | 0.45× |
| joint v1 (16 Indian tiles) | 0.7661 | 0.369 | 0.546 | 1.28× |
| joint v3 (103 Indian tiles) | 0.7662 | 0.390 | 0.560 | 1.28× |
| OSM fine-tune | 0.6708 | **0.716** | **0.149** | 1.37× |
| envelope fine-tune | 0.6828 | — | — | 1.59× |

The envelope fine-tune's recall columns are left blank on purpose: it predicts
building-cluster envelopes, so a "footprint recall" against OSM footprints is
not the same measurement as everyone else's and putting it in the same column
would invite exactly the comparison `label_semantics` exists to prevent.

Inria is pooled IoU over the 25 official val tiles. Bangalore is 141 of 142
OpenStreetMap ways at threshold 0.50 — **recall only**; the precision column is
omitted on purpose, because it measures OpenStreetMap's incompleteness rather
than any model (see the three-way comparison above).

### Finding 1 — more Western data buys Inria and costs India

Doubling `samples-per-tile` from 48 to 96 draws twice as many training windows
from the same 155 Inria tiles. It is the cleanest experiment in the project:
one variable, same architecture, same encoder, same schedule.

| | shipped (spt 48) | control (spt 96) |
|---|---|---|
| Inria pooled IoU | 0.7712 | **0.7843** (+0.0131) |
| Bangalore footprint recall | 0.241 | 0.191 |
| silent fraction | 0.560 | **0.716** |
| detected area ÷ OSM | 0.70× | 0.45× |

More data from five Western cities produced a *more Western* model. It is the
best Inria checkpoint the project has ever trained and it is the worst of the
five in Bangalore, where it goes silent on 72% of buildings a human mapped.
It is registered in `model/manifest.json` and is **not** the default.

This is the sharpest available evidence that the general model cannot be grown
into a global one with more Inria.

### Finding 2 — joint training fixed forgetting, not the domain gap

Joint training did exactly what it was designed to do. Fine-tuning on ~100
narrow tiles costs about 0.10 pooled Inria IoU; joint training costs **0.005**.

| | Inria pooled IoU | vs shipped |
|---|---|---|
| shipped | 0.7712 | — |
| joint v3 | 0.7662 | **−0.0050** |
| OSM fine-tune | 0.6708 | −0.1004 |

But it recovered only about a third of the Indian gain that fine-tuning on the
*same 103 tiles* achieves, and it left the failure mode untouched:

| | shipped | joint v3 | OSM fine-tune |
|---|---|---|---|
| footprint recall | 0.241 | 0.390 | **0.716** |
| silent fraction | 0.560 | 0.560 | **0.149** |
| Bangalore IoU | 0.254 | 0.242 | **0.409** |

The silent fraction is the tell. Joint v3 raised recall **without reducing
silence at all** — it did not make the model see more buildings, it made it
paint more area around the ones it already saw. Its Bangalore IoU did not
improve, its detected area rose to 1.28× the OSM reference, and its Bangalore
precision fell to 0.347 where every other model scores ~0.49. Recall by size is
incoherent with a real improvement: 0–50 m² rose (0.083 → 0.500) while the
200–500 m² band *regressed* (0.462 → 0.231).

### Finding 3 — how big a difference has to be before it means anything

There are two joint checkpoints, `joint_effb0_20260905.pt` and
`joint_v3_effb0_20260906.pt`. They are different files with different weights.
**They are also, as far as anything in this repo can show, two executions of the
same configuration** — the committed training log and the one downloaded on
2026-09-06 have the same epoch size (~594 steps, 367 s/epoch), the same training
loss to four decimals at every epoch, the same 46 epochs, and `best_val_iou`
differing by 7e-6 (0.7174775 vs 0.7174843).

That got checked because this section was going to claim something else. The
handoff notes record that an early joint run trained on only 16 of the 103
Indian tiles, because Kaggle mounted the dataset while it was still processing —
a real incident, and the reason `kaggle_joint/` now asserts the tile count. The
tempting story was "16 tiles vs 103 tiles changed nothing". **The artifacts do
not support it.** No 16-tile training log survives here; the one committed
alongside the fix is a full-data run, and which run produced
`joint_effb0_20260905.pt` can no longer be established.

So read the pair as what it defensibly is — a **repeat-run noise estimate** for
these two harnesses, which is worth more than the claim it replaced:

| | run A | run B | spread |
|---|---|---|---|
| Inria pooled IoU | 0.7661 | 0.7662 | 0.0001 |
| Bangalore pixel recall | 0.460 | 0.443 | 0.017 |
| footprint recall | 0.369 | 0.390 | 0.021 |
| silent fraction | 0.546 | 0.560 | 0.014 |
| area / OSM | 1.284 | 1.276 | 0.008 |

**Inria is reproducible to ~0.0001; the Bangalore numbers wobble by ~0.02.**
That is the bar. Joint's gain over the shipped model (footprint recall
0.241 -> 0.390) is roughly seven times the noise and is real. A difference of
0.02 between two Indian numbers is not interpretable, and one run is not
evidence of a small effect.

The open question stands where it did: joint training under-learns the Indian
half, and the `--extra-repeat 20` weighting that makes 103 tiles ~22% of an
epoch is the untested variable. Change the weighting or stage the training, and
judge it against a bar of 0.02.

### The decision: the default does not change

`unetpp_effb0_inria_20260903.pt` stays `"default": true`.

* The **B0 control** is better on Inria and worse everywhere this project wants
  to go. Promoting it would improve the headline number and degrade the product.
* **Joint v3** is the most interesting result and still not promotable. It buys
  0.149 of footprint recall for an 82% increase in detected area that cannot be
  verified — OpenStreetMap's incompleteness means "more area" and "more correct
  area" are indistinguishable here — in an app that multiplies area into money.
  Its unchanged silent fraction says the extra area is not new buildings.
* **`finetune_osm`** remains the right choice for India and stays opt-in behind
  `RSOLAR_MODEL=finetune_osm`.

### A correction to an earlier number

The "fine-tuning costs ~0.16 Inria IoU" figure came from a **5-tile** subset.
Measured on the full 25-tile official val split it is **0.10** (0.7712 →
0.6708). The conclusion is unchanged and the magnitude is smaller; the 5-tile
subset should not be used again now that the full split is cheap to run.

### Why B3 never produced a number

`kaggle_inria_b3` was cancelled during its first epoch, and it could not have
finished anyway. Its `status.json` records `amp: false` — it *did* pass
`--amp fp16`, but `utils.select_amp` NaN-probes whatever it is asked for and
EfficientNet-B3 failed the probe on the P100, so it fell back to fp32. At the
recorded 8.86 images/s over 1240 batches of 8, one epoch is ~18.7 minutes and
50 epochs is **~15.5 h** against Kaggle's 12 h limit.

An explicit `--amp` request selects the *candidate list*; it does not skip the
safety probe. On hardware where the probe fails, an explicit request is not a
guarantee, and the run silently becomes twice as slow as it was budgeted for.

### Published

Both checkpoints are on GitHub as
[`v1.1-round2`](https://github.com/Parthesh10/rooftop-solar-potential-detection/releases/tag/v1.1-round2)
(2026-09-06), each as `.pt`, `.onnx` and sidecar. The shipped model and the two
fine-tunes remain on `v1.0-inria`.

```bash
gh release download v1.1-round2 --dir webapp/models
RSOLAR_MODEL=joint_v3_effb0_20260906 python -m webapp
```

Both sidecars carry `"default": false`. Exporting them into `webapp/models/`
alongside the shipped model was the first real test of the fix for the
"newest .onnx wins" trap, and it held: with two *newer* `.onnx` files present,
`load_model()` still returns `unetpp_effb0_inria_20260903` because selection
reads the `default` flag before falling back to mtime. Under the old rule this
release would have silently swapped every user's model.

---

## The default changes: joint v2, ten Indian cities (2026-09-07)

**`joint_v2_india10_20260907.pt` replaces `unetpp_effb0_inria_20260903.pt` as the
shipped model.** It is the first model this project has produced that improves
India without paying for it on Inria. Every previous attempt bought one at the
other's expense — that trade is what facts 30 and 31 are about.

### What changed in the training data

`kaggle_joint_v2/`: Inria plus **505 Indian tiles from ten cities**, repeated 8×
so India is ~35% of each epoch. The previous joint run had 103 tiles from two
cities at ~22%. 402 of the new tiles were machine-labelled from Google Open
Buildings across eight cities that had never appeared in training — Delhi,
Mumbai, Chennai, Kolkata, Hyderabad, Jaipur, Ahmedabad, Pune — with no human
labelling (fact 34).

### The measurement that decided it, and the one that nearly misled

The Bangalore block said the experiment had **failed**: footprint recall
0.390 → 0.333, silent fraction 0.560 → 0.589 against joint v3. Two things were
wrong with reading that as the answer.

**First, the Bangalore number is contaminated for every joint model.**
`bangalore_cvraman_a` fully contains the CV Raman Nagar benchmark block, all 30
of its tiles are in the joint training set, and `train_inria.py
--extra-data-dir` holds *nothing* back — validation is Inria-only. Both joint v3
and joint v2 trained on the block they were scored on. Only `finetune_osm`, which
used `--holdout-aoi bangalore_cvraman_a`, has ever had a clean Bangalore figure.

**Second, and more important: Bangalore could not see the experiment.** v2's
whole thesis was geographic breadth, and Bangalore tiles were already in v3's
training set. The 402 new tiles taught the model eight cities the Bangalore
benchmark says nothing about. **Measuring a breadth experiment at a single point
can only show its cost, never its benefit.**

### Held-out blocks in six of the new cities

Each block is disjoint from every training AOI. OSM-anchored **recall only** —
OSM is incomplete in India, so precision and IoU there measure the reference
(fact 28).

| city | OSM refs | shipped recall / silent | **joint v2 recall / silent** |
|---|---|---|---|
| Chennai | 242 | 0.302 / 0.379 | **0.872 / 0.081** |
| Pune | 197 | 0.202 / 0.596 | **0.839 / 0.083** |
| Delhi | 130 | 0.168 / 0.744 | **0.840 / 0.080** |
| Ahmedabad | 38 | 0.103 / 0.793 | **0.414 / 0.517** |
| Mumbai | 30 | 0.346 / 0.538 | **0.769 / 0.154** |
| Jaipur | 18 | 0.059 / 0.706 | **0.353 / 0.294** |

Six of six, every one far outside the 0.02 noise floor. Kolkata and Hyderabad
are **unmeasured** — Overpass rate-limited those two blocks on three attempts.

Pune is the single most convincing row: IoU 0.163 → **0.493** and precision
0.553 → **0.593**. When precision and recall rise together against the same
reference, the reference's incompleteness cannot be the explanation.

### The over-painting question, settled in the West

v2 predicts 2.2× the OSM area in Chennai and 6.3× in Delhi. That looks alarming
and it is why joint v3 was not promoted. But OSM claims only **7.3%** of the
Delhi block is roof, where dense Indian urban is realistically 40–60%; converted
to physical coverage, v2 says 46% and the shipped model says 11.5%.

The way to settle it is to measure somewhere OSM is complete:

| block | model | IoU | precision | recall | area ÷ OSM |
|---|---|---|---|---|---|
| **Austin** | shipped | 0.814 | 0.898 | 0.943 | **1.00×** |
| | **joint v2** | 0.797 | 0.896 | 0.962 | **0.98×** |
| **Vienna** | shipped | 0.482 | 0.502 | 0.940 | 1.84× |
| | **joint v2** | 0.486 | 0.503 | 0.940 | 1.86× |

In Austin, where the shipped model reproduces OSM's area to 1.00×, **v2 predicts
0.98× at the same precision**. It is not an over-painting model. The Indian
ratios are OpenStreetMap's incompleteness — argued before, proven here.

### The cost

| | shipped | joint v2 |
|---|---|---|
| Inria pooled IoU @0.50 | 0.7712 | **0.7690** |
| per-window mean | 0.7233 | 0.7219 |

**0.0022**, an order of magnitude inside the gap between the fine-tunes (0.10)
and roughly the reproducibility of the metric itself.

### What this does and does not change

* **Indian area estimates roughly double.** The coverage analysis and the Austin
  result both say that is more correct, not less, but it is a large change to
  every kWh and money figure in those regions.
* **Building count is less reliable than area** — v2 merges adjacent roofs into
  blobs in dense blocks. Visible in `compare_chennai.png`.
* **The regional threshold prior is now nearly vestigial.** `coverage.py` relaxes
  to 0.40 outside the training cities because the old model was under-confident
  there. For v2 that is worth +4% area in Chennai (2.24× → 2.33×) and no recall
  worth mentioning — it is no longer silent, so there is nothing to relax. Left
  in place; it is now a no-op rather than a fix.
* **0.7712 still belongs to the old model.** Quote it as such, not as the current
  default's score.

### Still missing

No hand-drawn evaluation set exists for any of the eight new cities. Everything
above is OSM-anchored recall, which is honest but cannot give an IoU. That is
the next thing worth a human's time.

---

## Round 3: four continents, and what the domain boundary actually is (2026-09-08)

`joint_v3_world25_20260908.pt` — Inria plus **1325 tiles from 25 AOIs on four
continents**. Designed as a single-variable experiment: the non-Western share of
each epoch is held at **34.8%** against joint v2's 35.2% (1325 tiles × repeat 3
versus 505 × repeat 8), so the only thing that changes is *how many places*.

That design mattered. Joint v2 moved the tile count and the epoch share
together, so fact 36's attribution to breadth was inference rather than
measurement. This run tests it directly.

### Held-out blocks in the new continents

Every block is disjoint from all 25 training AOIs. OSM-anchored **recall** —
precision and IoU against OSM measure the reference, not the model (fact 28).
Each cell is Inria-only → joint v2 → round 3.

| block | refs | footprint recall | silent fraction | IoU |
|---|---|---|---|---|
| Manila | 937 | 0.495 → 0.692 → **0.964** | 0.267 → 0.140 → **0.010** | 0.419 → 0.476 → **0.553** |
| São Paulo | 932 | 0.151 → 0.461 → **0.886** | 0.683 → 0.366 → **0.045** | 0.174 → 0.353 → **0.496** |
| Jakarta | 705 | 0.438 → 0.763 → **0.913** | 0.287 → 0.102 → **0.037** | 0.290 → 0.399 → **0.423** |
| Lagos | 624 | 0.058 → 0.579 → **0.796** | 0.861 → 0.245 → **0.096** | 0.110 → 0.373 → **0.440** |
| Nairobi | 557 | 0.162 → 0.523 → **0.750** | 0.704 → 0.359 → **0.105** | 0.315 → 0.375 → **0.408** |
| Lima | 289 | 0.157 → 0.821 → **0.905** | 0.675 → 0.066 → **0.026** | 0.160 → 0.335 → **0.354** |

Monotonic in six of six, on both metrics, with 289–937 human-drawn references
per block. São Paulo is the single strongest result the project has measured:
IoU nearly triples **and precision rises** (0.460 → 0.527) while recall goes
0.151 → 0.886. Recall gains almost always cost precision; both improving on the
largest sample rules out "it just paints more area" for that block.

### The finding nobody was looking for

Read the **middle** column. Joint v2 trained on India and nothing else, yet:

| block | Inria-only | joint v2 (India only) |
|---|---|---|
| Lima | 0.157 | **0.821** |
| Lagos | 0.058 | **0.579** |
| Nairobi | 0.162 | **0.523** |
| São Paulo | 0.151 | **0.461** |

Five times better in Lima, ten times in Lagos, in continents it had never seen a
pixel of.

**So the domain boundary is not national or continental.** What separates these
images from Inria is not "India" — it is dense, small-plot, high-contrast urban
form, which São Paulo and Lagos and Jakarta share with Chennai and do not share
with Austin. Facts 30 and 31 framed the problem as *India versus the West*; it
was always *dense Global South versus sparse Western*, and India was simply the
first sample of it anyone collected.

That has a practical edge: **each new region is cheaper than the last**, because
it buys transfer to its neighbours in built form rather than in geography.

### India, where no new tiles were added

Chennai, Delhi and Pune gained nothing new in round 3 — the eight new Indian
AOIs are tier-2 cities. Any movement here is those cities generalising sideways
to metros they have never seen:

| block | footprint recall | silent |
|---|---|---|
| Chennai | 0.872 → **0.915** | 0.081 → **0.043** |
| Delhi | 0.840 → **0.912** | 0.080 → **0.064** |
| Pune | 0.839 → **0.870** | 0.083 → 0.083 |

Small, but in the right direction on five of six figures, and consistent with
the transfer story above.

### The cost, three rounds running

| model | Inria pooled IoU @0.50 | precision |
|---|---|---|
| Inria-only (2026-09-03) | 0.7712 | 0.8454 |
| joint v2, ten Indian cities | 0.7690 | 0.8477 |
| **round 3, 25 AOIs / four continents** | **0.7671** | 0.8428 |

**0.0041 total** for four continents of coverage. And note the precision column:
0.8454 / 0.8477 / 0.8428 is flat, measured against Inria's *ground-truth* labels
rather than OSM. That is direct evidence of no Western over-painting, and it is
stronger than the Austin check because the labels are complete by construction.

Training IoU tells the same story from the other side — 0.838 → 0.804 → 0.755
across the three models. The training set is getting genuinely harder as it
diversifies; the model is not degrading.

### Still missing

* **No hand-drawn eval set for any of the 25 regions.** Every non-Inria number
  here is OSM-anchored recall. It is honest and it is the right metric given an
  incomplete reference, but it cannot produce a trustworthy IoU.
### The over-painting check, settled in the West

Round 3 predicts 1.5–2.0× the OSM area in the new continents, which is the
figure that would normally block a promotion. Austin answers it, because there
OSM is complete enough that precision and area both mean something:

| Austin | IoU | precision | recall | silent | area ÷ OSM |
|---|---|---|---|---|---|
| Inria-only | 0.814 | 0.898 | 0.943 | 0.019 | 1.00× |
| **round 3** | **0.828** | 0.891 | **0.962** | **0.010** | **1.03×** |

Round 3 does not merely avoid regressing in the West — it is **better** there
than the model trained on nothing but Western cities, against complete ground
truth, while tripling recall across the Global South.

Inria's own ground-truth labels say the same from the other direction:
precision 0.8454 / 0.8477 / 0.8428 across the three models is flat.

**So this model is promoted to the default.**

### A verification guard that had it exactly backwards

`scripts/export_onnx.py` refused to ship this model: `max |torch - onnx| =
1.25e-03 (TOO LARGE)`. Investigating rather than overriding it turned up a real
bug in the check.

It compared raw **logits**, on a fresh `torch.randn` draw, against an absolute
1e-3 tolerance. Measured across five seeds and against real imagery:

| | noise logits (what the guard tested) | real imagery, probability error | decision flips |
|---|---|---|---|
| joint_v3 (**failed**) | 7.4e-04 – 2.0e-03, failed 4/5 draws | **8.7e-05** | **2** / 1,048,576 |
| joint_v2 (**passed**) | 9.9e-05 – 1.2e-04, passed 5/5 | 4.2e-04 | 3 / 1,048,576 |

**The guard blocked the better export and had already shipped the worse one.**
On real imagery joint_v3's probability error was five times *smaller* than the
model it passed, and it flipped fewer pixels.

The discrepancy lives where the sigmoid is saturated. A 1.6e-03 wobble at
|logit| ≈ 15 moves the probability by ~1e-8 and cannot change an output. The
guard measured error exactly where the model is most certain — where error
cannot matter — and a model that generalises more broadly saturates harder on
out-of-distribution noise, so **the better the model, the more likely it was to
be rejected**.

The check now uses a fixed seed, an input scaled like normalised imagery, and
gates on the **probability** error plus the fraction of thresholded decisions
that change. All three shipped models pass with zero decision flips:

| model | logit diff | probability diff | flips |
|---|---|---|---|
| joint_v3 | 1.63e-03 | 1.10e-08 | 0 |
| joint_v2 | 1.34e-04 | 5.60e-10 | 0 |
| Inria-only | 5.91e-05 | 2.18e-09 | 0 |

---

## The first real IoU outside Inria: ramp Karnataka (2026-09-08)

Every non-Inria number above this line is OSM-anchored **recall**, because
OpenStreetMap is incomplete wherever this model is interesting. That is the
honest metric for an incomplete reference, but it cannot produce a trustworthy
IoU or precision, and after round 3 not one of the 25 training regions had any
human-drawn ground truth at all.

[**ramp**](https://source.coop/ramp/ramp) (Replicable AI for Microplanning,
CC-BY-NC-4.0) closes that gap, and is a better fit than the obvious alternatives
on three counts simultaneously:

* **The labels are rooftops, not ground footprints.** ramp's own Shanghai README
  says the SpaceNet labels were revised "to be consistent with the ramp datasets
  notion of rooftop as the building footprint". Everything this project has used
  — Inria, OSM, Open Buildings — labels ground footprints. This app turns *roof*
  area into kWh, so ramp's semantics are the ones it actually wants.
* **It is human-reviewed and exhaustive.** "Tier 1" means thoroughly reviewed
  and improved. Because the tiles are exhaustively labelled, **precision and IoU
  mean what they say here** — a false positive is a real false positive, not an
  unmapped building.
* **It is 30 cm.** Inria is 0.30 m/px and the app serves Esri z19 at ~0.30, so
  nothing is rescaled. Most candidates fail this: Open Cities is 2–20 cm drone
  imagery. (ramp's Accra set *is* Open Cities, resampled to 30 cm.)

`scripts/download_ramp.py` fetches any of the 22 regional datasets from open S3;
`scripts/eval_ramp.py` scores checkpoints against them, reading the GeoTIFF
tie-point and pixel-scale tags directly rather than adding GDAL or rasterio.

### The result: 494 tiles, 3,945 human-drawn buildings, rural Karnataka

11.7% of the reference area is roof — sparse compared with the dense urban
blocks every other benchmark in this file uses.

| model | IoU | precision | recall | area ÷ actual |
|---|---|---|---|---|
| Inria-only (2026-09-03) | 0.150 | 0.876 | 0.153 | 0.17× |
| joint, 103 tiles / 2 cities (09-06) | 0.298 | 0.796 | 0.322 | 0.41× |
| joint v2, 505 tiles / 10 cities (09-07) | 0.469 | 0.731 | 0.568 | 0.78× |
| **joint v3, 1325 tiles / 25 AOIs (09-08)** | **0.579** | 0.727 | **0.739** | **1.02×** |

**Monotonic in four of four, and the IoU quadruples.**

### Why this matters more than another recall table

This is an **independent audit of every promotion decision the project made**, on
data that shares nothing with how those decisions were reached: different labels
(human, exhaustive, rooftop semantics), different sensor (Maxar, not Esri),
different built form (rural and agricultural, absent from all 25 training
regions), and never seen in training.

It produces **exactly the same model ordering** the OSM-anchored recall numbers
did at every step. That retroactively validates the OSM-recall methodology
itself — the metric that had to be taken on trust because an incomplete
reference cannot give an IoU.

The last column is the one that reaches users. The app multiplies roof area into
kWh and money, and the original model recovered **17% of the actual roof area**
in rural Karnataka — it would have under-estimated solar potential there roughly
six-fold. The current default lands at **1.02×**. Together with Austin's 1.03×
(complete OSM, Western), the app's area estimates are now confirmed against
complete ground truth in two very different places.

Precision falls 0.876 → 0.727 as recall rises 0.153 → 0.739. That is the normal
trade and it is a good one: the old model's high precision was the arithmetic of
finding almost nothing.

### Caveats, stated rather than buried

* **Maxar imagery, not Esri.** Same 30 cm, different sensor and processing, so
  part of any residual gap is provider domain shift rather than model quality.
* **The 256 px tiles are reflect-padded to the model's 512 window.** Measured
  rather than assumed: eroding a 32 px border changes IoU by 0.004–0.026, and
  in the *favourable* direction for the current model (0.579 → 0.604).
* **CC-BY-NC-4.0.** Non-commercial, and more restrictive than Inria, Open
  Buildings (CC-BY) or OSM (ODbL). Fine for research; a deliberate decision
  would be needed before commercial use. This is why ramp is used here as an
  **evaluation** set and not folded into training.

---

## The human-truth benchmark, extended to four regions (2026-09-09)

`scripts/eval_ramp.py` now covers four ramp regions — **1,494 scored tiles and
26,946 human-drawn, exhaustively labelled rooftops**. Karnataka is held out of
training by construction; the other three are *in* the current default's
training set, which makes them a different and useful test.

| region | tiles | buildings | roof % | Inria-only | joint v2 | **joint v3 (default)** |
|---|---|---|---|---|---|---|
| Karnataka — rural, **held out** | 494 | 3,945 | 11.7 | 0.150 | 0.469 | **0.579** |
| Nairobi — urban/peri-urban | 400 | 8,053 | 18.1 | 0.523 | 0.544 | **0.666** |
| Accra — urban, dense | 400 | 11,613 | 40.7 | 0.389 | 0.408 | **0.525** |
| Dhaka — very dense urban | 200 | 3,335 | 29.4 | 0.375 | **0.401** | 0.318 |

The numbers are stable against sample size: Accra and Nairobi were scored twice,
at 200 and 400 tiles, and agreed to within 0.004 IoU (0.528 vs 0.525, 0.665 vs
0.666). That is worth knowing before reading any single row as precise.

**Three of four go the right way, decisively, and the promotion stands.** The
strongest single row is Karnataka, because it is the one region no model has
ever trained on.

### Dhaka is a real regression, and it is not a threshold artifact

Swept across the whole usable range, joint v2 beats the default at **every** cut
point:

| threshold | v2 IoU | v3 IoU |
|---|---|---|
| 0.30 (best for both) | **0.4715** | 0.4142 |
| 0.50 | 0.4479 | 0.3918 |
| 0.65 | 0.4244 | 0.3732 |

Two things make it worth understanding rather than dismissing.

**Adding Dhaka to training made Dhaka worse.** It was absent from v2's ten
Indian cities and present in v3's 25 AOIs (56 tiles), and it got worse. Two
candidate explanations, neither yet tested: the 56 training tiles are **Esri**
imagery while this benchmark is **Maxar**, so what was learned may not transfer
across providers even at matched resolution; or dense-urban capacity was diluted
when the same 34.8% epoch share was spread across 25 AOIs instead of 10.

**Every model under-predicts badly there** — 0.47x to 0.72x of the real roof
area even at the loosest threshold, against 1.02x in Karnataka. That is a shared
weakness in very dense low-rise fabric that the breadth strategy did not fix,
not something v3 introduced.

### This is the first time the two benchmarks have disagreed

OSM-anchored recall said v3 improved in six of six held-out blocks. The human
benchmark says three of four. Both are honest; they measure different things,
and the disagreement is localised to one region.

That is the argument for having built it. An incomplete reference can only
support recall, and recall rewards a model for predicting more. Dhaka is
precisely where that distinction bites: v3 predicts *less* area there than v2
(0.47x against 0.58x), which an OSM recall metric would have shown as a
difference and an exhaustive IoU shows as a regression.
