# Kaggle GPU training

    python -m kaggle kernels push -p kaggle --accelerator gpuT4x2
    python -m kaggle kernels status partheshgupta/rooftop-solar-u-net-training
    python -m kaggle kernels output partheshgupta/rooftop-solar-u-net-training -p kaggle_out

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
