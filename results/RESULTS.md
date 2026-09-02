# Results

Two datasets, two sets of numbers. **Inria is the headline** — it is what the
shipped general model is trained and scored on. The Swiss DOP25 numbers below it
are the project's history and a secondary out-of-distribution check.

---

# THE SHIPPED MODEL — Inria, 2026-09-03

**Inria official val IoU 0.7233.** Target was ≥ 0.72. ✅

| | |
|---|---|
| Weights | `results/unetpp_effb0_inria_20260903.pt` (26 MB) |
| Architecture | **U-Net++ / EfficientNet-B0**, ImageNet-pretrained encoder |
| Parameters | 6.6 M |
| Trained on | Inria official split — 155 tiles (austin, chicago, kitsap, tyrol-w, vienna), tiles 6–36 per city |
| Scored on | Inria official val — 25 tiles, tiles 1–5 per city, **never seen in training** |
| Input | 512×512 @ 0.3 m/px, ImageNet normalisation |
| Labels | **building footprints** — roof extent, not installable area |

| metric | value |
|---|---|
| **IoU** | **0.7233** |
| F1 | 0.8180 |
| accuracy | 0.9634 |
| precision | 0.8226 |
| recall | 0.8568 |

Config: `--window 512 --samples-per-tile 48 --pos-weight 2.4 --dice-weight 0.6
--epochs 60 --batch-size 16 --lr 3e-4 --patience 12`, AdamW + cosine with
5-epoch warmup, D4 + photometric augmentation. Kaggle T4, fp16, ~4.8 h/config.

## The Inria run

| run | arch / encoder | best epoch | **val IoU** | F1 | P | R |
|---|---|---|---|---|---|---|
| **I2** | **U-Net++ / efficientnet-b0** | 49 | **0.7233** | 0.818 | 0.823 | 0.857 |
| I1 | U-Net / resnet34 | 59 | 0.7178 | 0.813 | 0.824 | 0.850 |

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
| 2026-09-03 | **U-Net++ / effb0** | **Inria val** | **0.7233** | **shipped model** |
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
