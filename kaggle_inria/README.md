# Kaggle — Inria training

Trains the **general** rooftop model on the full Inria dataset. Separate kernel
from `kaggle/` (which is the Swiss encoder sweep).

    git push origin main            # ALWAYS FIRST — the notebook clones from GitHub
    python -m kaggle kernels push -p kaggle_inria --accelerator gpuT4x2
    python -m kaggle kernels status partheshgupta/rooftop-solar-inria-training
    python -m kaggle kernels output partheshgupta/rooftop-solar-inria-training -p kaggle_inria_out

## No dataset upload

Inria (22 GB) is mounted from the public Kaggle copy
`sagar100rathod/inria-aerial-image-labeling-dataset` via `dataset_sources` in
`kernel-metadata.json`. It appears under `/kaggle/input/...`; cell 1 locates the
`AerialImageDataset/` directory rather than hardcoding the path. The repo is
still `git clone`d for the code, so **push to GitHub before pushing the kernel.**

## What it runs

`scripts/train_inria.py` on Inria's official split (tiles 1–5 → val, 6–36 →
train; 155 / 25 tiles). Two architectures carried over from the Swiss sweep:

    I1_unet_rn34     U-Net   + ResNet34/ImageNet
    I2_unetpp_effb2  U-Net++ + EfficientNet-B2/ImageNet

512×512 windows read on the fly from the 5000² GeoTIFFs
(`process_data.inria.InriaWindowDataset`), `samples_per_tile=48` → ~7.4k
windows/epoch, 60 epochs. Estimate ~1.5–3 h on one T4 — inside the 12 h cap.

## If a session times out

Checkpoints are written every 5 min and every epoch. Re-push and the notebook
resumes only if the previous `runs/` survives — it does **not** across a fresh
kernel run, so for a genuine multi-session resume, pull the run's `state.pt`
from `kernels output` and add it back as an input, or lower `samples_per_tile` /
`epochs` so one session finishes. First runs so far complete in one session.

## Status

Round 1 (2026-09-03) **completed**: U-Net++/EfficientNet-B0 and U-Net/ResNet34,
60 epochs each, ~4.8 h/config. effb0 won and is the shipped model — pooled Inria
val IoU **0.7712**. Artifacts are in `results/`.

`_gen_notebook.py` now holds a **staged, not-yet-run round 2**: EfficientNet-B3
against an EfficientNet-B0 control, `--data-parallel` across both T4s, ~10 h
total. It tests whether the remaining gap to published work (0.78–0.82) is model
capacity. Pushing this kernel will start that run — edit `SWEEP` first if you
want something else.

## Target

Inria official val IoU **≥ 0.72** (`plan.md` §4.3). Published building-seg work
sits ~0.78–0.82.

## Round 2 was split in two (2026-09-05)

The original round 2 ran **13 h and was cancelled** at the 12 h session limit.
Two causes, both only visible after the fact:

1. **Kaggle handed out a single P100, not the 2x T4 the push requested.**
   `--accelerator gpuT4x2` is a preference, not a guarantee. Half the compute,
   and the ~9.6 h estimate for two configs became ~13 h.
2. **Memory headroom was thinner than assumed.** The joint run peaked at
   **14.7 GB of 16 GB** with EfficientNet-B0 at batch 16 on that P100.
   EfficientNet-B3 at the same batch would not have fit.

So round 2 now lives in two kernels, each with a full 12 h session and its
epoch count untouched:

| kernel | encoder | samples/tile | batch | est. on one P100 |
|---|---|---|---|---|
| `kaggle_inria_b3/` | efficientnet-b3 | 64 | 8 | ~8.6 h |
| `kaggle_inria_b0/` | efficientnet-b0 | 96 | 16 | ~9.8 h |

Batch sizes are chosen to fit **one** 16 GB card rather than assuming the work
is split across two, and `--data-parallel` is now added only when
`torch.cuda.device_count() > 1`.

```powershell
git push origin main
python -m kaggle kernels push -p kaggle_inria_b3 --accelerator gpuT4x2
python -m kaggle kernels push -p kaggle_inria_b0 --accelerator gpuT4x2
```

Kaggle runs a limited number of GPU sessions at once; a third push queues
rather than failing.

### Sizing for a P100, measured rather than assumed (2026-09-05)

Both split kernels were also handed P100s. Two consequences, from real numbers
rather than estimates:

**AMP is off by default on this card.** `utils.select_amp` disables AMP on
anything without tensor cores, and a P100 is sm_60. That rule was written for a
GTX 1650, where emulated bf16 measured 2.6x slower than fp32 — but a P100 has
*native packed fp16* (~2:1 throughput) even though it has no tensor cores, so
the rule is too conservative here. Both kernels now pass `--amp fp16`
explicitly, which still NaN-probes against the real model and falls back to
fp32 if the probe fails.

**EfficientNet-B3 at 64 samples/tile does not fit a 12 h session.** From the
joint run's measured 0.757 s/step for B0 at batch 16, B3 at batch 8 works out
to ~1.04 s/step; 1240 steps/epoch x 50 epochs is ~18 h. Samples per tile drops
64 -> 28, which keeps all 50 epochs and brings the run to ~6 h. Note when
reading the result that B3 therefore sees fewer windows per epoch than round
1's B0 did (28 vs 48) — if B3 wins, it wins on less data, which strengthens the
capacity argument; if it loses, that is confounded and not conclusive.
