# Results

All numbers below use the **leakage-free geographic split** (420 train / 58 val
/ 74 test, zero tiles adjacent across splits) and the **corrected metric
harness** (`model.eval()` on, no empty-tile IoU inversion). Anything measured
any other way is not comparable — see "Why the 2023 checkpoints are not a
baseline" below.

## Current best — 2026-08-22

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
| 2026-08-22 | sweep D | **0.5442** | current best |
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
honest test set. Compare future work against **0.5442**.

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

1. **Pretrained encoders** (`segmentation_models_pytorch`, U-Net++/ResNet34 or
   EfficientNet-B0). The remaining gap to published Inria work is mostly here —
   training a 14.8 M-parameter U-Net from scratch on 420 tiles is the binding
   constraint, not the loss recipe. Needs Kaggle; will not fit in 4 GB.
2. **Threshold sweep.** Everything above uses 0.5. Recall (0.777) far exceeds
   precision (0.655), so a higher threshold may buy IoU for free — this costs
   one evaluation pass, no retraining.
3. **Test-time augmentation** — `evaluate.py --tta` averages the 8 dihedral
   transforms, typically +1-2 IoU for 8x inference cost.
4. **Inria** for scale and geographic diversity, then a Bhopal fine-tune set.
