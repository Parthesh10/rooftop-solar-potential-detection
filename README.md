# Rooftop Solar Potential Detection from Aerial Imagery

Binary pixel-wise segmentation of rooftop area available for photovoltaic (PV)
installation, using a U-Net on high-resolution aerial imagery.

Originally a B.Tech major project (MANIT Bhopal, 2022–23, Group 17). The code in
this repository is currently being audited and repaired; see
[`../potential-fixes.md`](../potential-fixes.md) and [`../plan.md`](../plan.md).

---

## ⚠️ Status of the reported results

**The metrics previously published in this README and in the project report are
not valid measurements.** They were produced by a harness with several defects
that are now fixed but whose effect on the numbers cannot be estimated after
the fact:

| Defect | Effect |
|---|---|
| `test_model()` never called `model.eval()` (F-03) | Every BatchNorm used *current-batch* statistics at batch size 2, so each image's prediction depended on whichever image shared its batch. |
| The IoU metric inverted empty tiles (F-08) | Any all-background tile the model got right scored a free **1.0**, inflating the mean by an unknown amount. |
| `recall()` divided by zero (F-11) | `nan` propagated into aggregates. Report Table 3 has Accuracy repeated in the Recall column, likely a hand-patched `nan`. |
| Train/test split was a random shuffle of adjacent tiles (F-05) | Train and test leak into each other, so the gap understates the true generalisation error. |

For the record, what was reported in 2023 (U-Net, 130 epochs, Swiss DOP25 set):

| Split | IoU | Accuracy | Recall | Precision |
|---|---|---|---|---|
| Train | 0.7524 | 0.9582 | 0.8629 | 0.8535 |
| Val | 0.6105 | 0.9196 | 0.7733 | 0.7693 |
| Test | 0.5566 | 0.9033 | 0.6952 | 0.7268 |

To establish the real baseline:

```bash
python evaluate.py --all --split all --out baseline.json
```

That output — not the table above — is what a retrained model should be
compared against. `model/manifest.json` has a `metrics_verified` field waiting
for it.

**On the report's V-Net and U-Net++ results:** no V-Net or U-Net++
implementation exists anywhere in this repository — only `model/unet.py` was
ever committed. Those figures cannot be reproduced from this code and should be
treated as unbacked until the models are rebuilt.

---

## Repository layout

```
config.py               paths, normalisation constants, geometry, TrainConfig
utils.py                device, seeding, pad_to_multiple / unpad
infer.py                THE inference path — preprocess, load_model, predict, TTA
evaluate.py             evaluation harness + CLI to re-score checkpoints
model/
  unet.py               vanilla U-Net, 3-ch in / 1-ch out, ~14.8M params
  manifest.json         per-checkpoint arch, normalisation, threshold, metrics
loss/
  loss.py               metrics: iou, f1, accuracy, recall, precision
  losses.py             objectives: Dice, Tversky, Combo, compute_pos_weight
process_data/
  data_loader.py        paired image/mask Dataset with D4 augmentation
  split.py              geographic / Inria-official / city-holdout splitting
  import_test.py        visualise a prediction on an arbitrary image
train/train.py          training loop with AMP, scheduling, checkpointing
plots/plots.py          training curves
tests/                  pytest regression suite
main.ipynb              driver notebook
```

## Setup

```powershell
uv venv --python 3.11
.venv\Scripts\activate
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

Swap the index URL for `.../whl/cpu` on a machine without an NVIDIA GPU.

Point `DATA_ROOT` at your dataset if it does not live in `./data`:

```powershell
$env:DATA_ROOT = "E:\...\dataset"
```

## Usage

```bash
# regenerate a leakage-free split (F-05)
python -m process_data.split --root data --out data/splits

# re-score the existing checkpoints under a correct harness
python evaluate.py --all --split all

# score one checkpoint with test-time augmentation
python evaluate.py --ckpt "model/path raise 130.pt" --split test --tta

# tests
pytest
```

Training runs from `main.ipynb`, or from your own script:

```python
from config import TrainConfig
from train.train import training_model, build_optimizer, build_scheduler
from loss.losses import build_loss

cfg = TrainConfig(epochs=80, lr=3e-4, batch_size=8)
loss_fn = build_loss(cfg, loader=train_loader, device=device)
opt = build_optimizer(model, cfg)
sched = build_scheduler(opt, cfg, steps_per_epoch=len(train_loader))

history = training_model(train_loader, loss_fn, opt, model,
                         num_epochs=cfg.epochs, scheduler=sched,
                         val_loader=val_loader, cfg=cfg, device=device)
```

Runs write `runs/<timestamp>/{best.pt, last.pt, metadata.json, history.json}`,
with the git SHA and full config recorded.

## Inference

`infer.py` is the only place preprocessing is defined. Do not reimplement it —
that divergence is what F-01 was.

```python
from infer import load_model, predict_mask

model, entry = load_model("model/path raise 130.pt")
mask = predict_mask(model, "my_area.png", tta=True)

GSD_M = 0.25                       # metres per pixel of the source image
area_m2 = mask.sum() * GSD_M ** 2
```

For images larger than one forward pass, `predict_large()` does Hann-weighted
sliding-window inference.

## Data

| Set | What | Where |
|---|---|---|
| Swiss DOP25 | 525 crops, 250×250 @ 0.25 m/px, ARA-style labels | `data/` (train 420 / val 52 / test 53) |
| Inria Aerial Image Labeling | 180 tiles, 5000×5000 @ 0.3 m/px, **building-footprint** labels | `../dataset/AerialImageDataset/` |

Two things to keep straight:

1. Inria's test cities (bellingham, bloomington, innsbruck, sfo, tyrol-e) ship
   **without ground truth**. Score on a held-out slice of *train* using
   `--inria` (tiles 1–5 of each city) or `--holdout-city`.
2. Inria labels are **building footprints**, not available rooftop area. A model
   trained on them predicts roof extent; converting that to installable area
   needs a packing factor for setbacks, walkways, tanks and inter-row spacing.
   Say "estimated" in any user-facing output.

## Known limitations

- Trained on Swiss / US / European rooftops. Indian rooftops (flat RCC, dense,
  water tanks, heavy shadowing) are a substantial domain shift and the model has
  not been evaluated on them.
- No obstruction-level segmentation: chimneys, skylights and existing panels are
  not separated out.
- Accuracy is a misleading headline metric here — the positive-pixel rate is
  roughly 10%, so predicting all-background already scores ~0.90. Read IoU.

## References

- Ronneberger, Fischer & Brox, *U-Net: Convolutional Networks for Biomedical
  Image Segmentation*, 2015
- Maggiori et al., *Inria Aerial Image Labeling Dataset*, 2017 —
  <https://project.inria.fr/aerialimagelabeling/>
- Zhou et al., *UNet++: Redesigning Skip Connections*, 2020
- Base U-Net implementation adapted from
  <https://github.com/hiyouga/Image-Segmentation-PyTorch>
