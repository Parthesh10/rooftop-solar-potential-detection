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

## Target

Inria official val IoU **≥ 0.72** (`plan.md` §4.3). Published building-seg work
sits ~0.78–0.82.
