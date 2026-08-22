# First trustworthy result — 2026-08-22

Trained on Kaggle (Tesla P100, 4.6 min, 7 s/epoch — about 12x the local
GTX 1650's 75-102 s/epoch) using the leakage-free geographic split and the
corrected metric harness.

Weights: `results/unet_swiss_geo_20260822.pt` (best epoch 12).

## Headline

| split | IoU | F1 | acc | precision | recall | n | undefined |
|---|---|---|---|---|---|---|---|
| train | 0.5845 | 0.7115 | 0.9222 | 0.6709 | 0.8339 | 420 | 12 |
| val   | 0.5466 | 0.6700 | 0.9278 | 0.6796 | 0.7511 | 58 | 7 |
| **test** | **0.4656** | 0.5736 | 0.8911 | 0.6404 | 0.6701 | 74 | 5 |

**Test IoU 0.4656 is the first number this project has produced that is
actually defensible.** Every earlier figure was measured either with the
broken harness (F-03 / F-08) or on a split that leaked (F-05), usually both.

## The 2023 checkpoint cannot be compared against this

Scoring `path raise 130.pt` on this same clean split gives test IoU 0.4946,
which looks better. It is not a valid comparison:

    new val  (58 tiles): 41 were in the 2023 training set  -> 71%
    new test (74 tiles): 51 were in the 2023 training set  -> 69%

The 2023 model was trained on a random split of the same 574 tiles, so roughly
seven of every ten tiles in *any* newly-drawn evaluation set are tiles it
already memorised. There is no way to re-partition this dataset that gives the
old checkpoint an honest test set. Its numbers — 0.5566 as reported in 2023,
0.5170 re-measured with the fixed harness, 0.4946 here — are all contaminated
to an unknown degree, and none of them are a baseline.

The only clean comparison would be a fresh model trained on the geographic
split, which is exactly what this run is. Future runs compare against 0.4656.

## The run under-trained

Early stopping fired at epoch 27 with the best epoch at 12, so:

* only ~12 epochs of useful training happened, against 80 requested;
* the cosine schedule never annealed — LR was still at 2.4e-4 when it stopped;
* val IoU swung between 0.375 and 0.54 across consecutive epochs, because the
  validation set is 58 tiles and therefore noisy. `patience=15` on a signal
  that noisy stops runs more or less at random.

Precision also collapsed relative to the old model (0.64 vs 0.78) while recall
rose (0.67 vs 0.60) — the estimated `pos_weight` of ~5.9 is pushing hard toward
over-prediction, which costs IoU.

## Next run

1. `--patience 40` (or disable early stopping) so the schedule completes.
2. Sweep `pos_weight` — try the sqrt of the imbalance (~2.4) and 1.0 against
   the current ~5.9, and `dice_weight` 0.5 vs 0.7.
3. Only then move to a pretrained encoder (plan.md 4.2), which is the change
   expected to actually move IoU rather than rebalance it.

At 4.6 min a run, each of these is cheap on Kaggle.
