# Results

All numbers below use the **leakage-free geographic split** (420 train / 58 val
/ 74 test, zero tiles adjacent across splits) and the **corrected metric
harness** (`model.eval()` on, no empty-tile IoU inversion). Anything measured
any other way is not comparable — see "Why the 2023 checkpoints are not a
baseline" below.

> **Direction (2026-08-29):** the Swiss set (420 train tiles) has been shown to
> be too small to benefit from a pretrained encoder — see the encoder sweep
> below. The project is moving to the full **Inria** dataset for a general
> model; Swiss becomes a secondary eval. New headline target: **Inria official
> val IoU ≥ 0.72**.

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

| date | model | test IoU | notes |
|---|---|---|---|
| 2026-08-29 | U-Net / resnet34 (ImageNet) | 0.5653 | Swiss encoder sweep; ties scratch |
| 2026-08-29 | U-Net / scratch (T4/fp16) | 0.5651 | same config as sweep D, +2 from hardware/noise |
| 2026-08-22 | sweep D (P100/fp32) | 0.5442 | Swiss baseline |
| 2026-08-22 | first clean run | 0.4656 | early-stopped at 27, best @ 12 |
| 2026-08-11 | 2023 `path raise 130.pt` | (0.4946) | **contaminated — not a baseline** |

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

1. **Inria training** — the main event now. `scripts/train_inria.py` +
   `kaggle_inria/` train on the full Inria set (official 1–5 split, 155/25
   tiles, on-the-fly 512² windows). Two archs: U-Net/resnet34 and
   U-Net++/efficientnet-b0. Target: Inria official val IoU ≥ 0.72.
2. **Broaden geography** — Inria is 5 Western cities. Add Google Open Buildings
   (Global South) and/or SpaceNet once the Inria pipeline is solid.
3. **Threshold + TTA** — cheap post-hoc gains on whichever model ships.
   `evaluate.py --tta` averages the 8 dihedral transforms (+1–2 IoU, 8× cost).
4. **Swiss as secondary eval** — keep scoring the shipped model on the Swiss
   geographic test split as an out-of-distribution check (different country,
   different label semantics, 0.25 m/px vs 0.3).
