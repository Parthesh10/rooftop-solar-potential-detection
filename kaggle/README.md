# Kaggle GPU training

    git push origin main            # ALWAYS FIRST — the notebook clones from GitHub
    python -m kaggle kernels push -p kaggle --accelerator gpuT4x2
    python -m kaggle kernels status partheshgupta/rooftop-solar-u-net-training
    python -m kaggle kernels output partheshgupta/rooftop-solar-u-net-training -p kaggle_out

## What the notebook does

Currently an **encoder sweep** (`_gen_notebook.py`): four architectures on the
Swiss geographic split with the loss recipe fixed at the sweep-D winner
(pos_weight 2.4, dice_weight 0.7).

    U0_scratch       verbatim 2023 U-Net (control)
    U1_unet_rn34     U-Net       + ResNet34/ImageNet
    U2_unetpp_effb0  U-Net++     + EfficientNet-B0/ImageNet
    U3_dlv3p_effb2   DeepLabV3+  + EfficientNet-B2/ImageNet

Cell 1 `pip install segmentation-models-pytorch`; encoder weights download from
the HF hub (needs `enable_internet`, same phone-verification gate as the clone).
`scripts/train_swiss.py` auto-switches normalisation to ImageNet stats for the
pretrained runs. Edit the `SWEEP` list in `_gen_notebook.py` and re-run it to
regenerate `kaggle_train.ipynb`, then push.

## Prerequisite: phone verification

**Kaggle gates both GPU/TPU accelerators and notebook internet access behind
phone verification.** Until the account is verified, a kernel can request a GPU,
the server will happily store the request, and the worker will still boot a
CPU-only image.

That is exactly what happened here on versions 1 and 2. Pulling the server-side
metadata back showed:

    "enable_gpu": true,
    "machine_shape": "Gpu"

while the worker reported:

    <nvidia-smi not found>
    torch 2.10.0+cpu | cuda available: False

Accepted-but-not-honoured means entitlement, not configuration. Verify at
kaggle.com -> Settings -> Phone Verification, then re-push.

This also matters because the notebook `git clone`s this repo, which needs
notebook internet — gated by the same verification.

## Accelerator flag is not optional

kaggle CLI 2.2.4 still reads `enable_gpu` from kernel-metadata.json, but the
server keys off `machine_shape` / `--accelerator`. Version 1 was pushed with
only `enable_gpu: true` and got a CPU worker.

## Notes

* The notebook clones the repo rather than mounting a dataset — the 169 MB
  Swiss set is tracked in-tree, so there is nothing to upload.
* The split is regenerated deterministically (seed 0), reproducing the local
  420 / 58 / 74 manifests with zero cross-split adjacency.
* The GPU governor is switched off for Kaggle (`--gpu-util-target 100`,
  `--gpu-temp-limit 0`): duty-cycling a datacenter card only burns quota.
