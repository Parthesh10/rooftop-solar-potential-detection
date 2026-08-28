"""Regenerate kaggle_train.ipynb.

The notebook source is written here rather than by hand because a stray "\n"
escape inside a cell string gets turned into a real newline by
``splitlines(keepends=True)``, which silently cuts a statement in half. Kernel
version 3 died that way with "unterminated f-string literal". Every generated
cell is therefore parsed with ``ast`` before the file is written.
"""

import ast
import io
import json

REPO_URL = "https://github.com/Parthesh10/rooftop-solar-potential-detection.git"


def md(t):
    return {"id": f"m{abs(hash(t)) % 99999}", "cell_type": "markdown",
            "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"id": f"c{abs(hash(t)) % 99999}", "cell_type": "code",
            "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


CELL_GPU = """import subprocess, sys, os, json, time, shutil
from pathlib import Path

def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return (r.stdout or r.stderr).strip()
    except FileNotFoundError:
        return "<" + cmd[0] + " not found>"

gpu_name = sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
print("GPU:", gpu_name)

if "not found" in gpu_name:
    raise SystemExit(
        "No GPU in this session. Push with: "
        "kaggle kernels push -p kaggle --accelerator gpuT4x2")

PROBE = (
    "import torch;"
    "p=torch.cuda.get_device_properties(0);"
    "cap='sm_'+str(p.major)+str(p.minor);"
    "t=torch.randn(256,256,device='cuda');"
    "ok=bool(torch.isfinite(t@t).all());"
    "print(torch.__version__, cap, cap in torch.cuda.get_arch_list(), ok)"
)

def probe():
    # Runs in a FRESH interpreter. That matters: this notebook process may
    # already hold a stale torch, and it is the subprocess view that decides
    # whether training works - scripts/train_swiss.py is itself launched as a
    # subprocess further down.
    r = subprocess.run([sys.executable, "-c", PROBE],
                       capture_output=True, text=True)
    return r.stdout.strip(), r.returncode, r.stderr.strip()[-400:]

out, rc, err = probe()
print("probe:", out or err)

# A P100 is sm_60 and the stock Kaggle image ships torch built for sm_70+.
# On that pairing torch.cuda.is_available() returns True while every kernel
# launch fails, so the arch list has to be checked explicitly, not trusted.
if rc != 0 or "True True" not in out:
    print("this torch cannot drive this GPU - installing a cu118 build")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "torch==2.4.1+cu118", "torchvision==0.19.1+cu118",
         "--index-url", "https://download.pytorch.org/whl/cu118"],
        capture_output=True, text=True)
    print(r.stdout[-1200:] if r.returncode == 0 else r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("cu118 torch install failed - see pip output above")
    out, rc, err = probe()
    print("probe after reinstall:", out or err)

if rc != 0 or "True True" not in out:
    raise SystemExit(
        "GPU still unusable after the reinstall (" + (out or err) + "). "
        "Request a T4 instead: --accelerator gpuT4x2")

print("GPU verified by real matmul in a fresh interpreter")"""


CELL_CLONE = """REPO = "__REPO_URL__"
WORK = Path("/kaggle/working")
SRC = WORK / "repo"

if SRC.exists():
    shutil.rmtree(SRC)
subprocess.run(["git", "clone", "--depth", "1", REPO, str(SRC)], check=True)
os.chdir(SRC)
sys.path.insert(0, str(SRC))
print("cloned at", sh(["git", "rev-parse", "--short", "HEAD"]))
print("train images:", len(list((SRC / "data" / "train" / "images").glob("*.png"))))

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "tifffile", "nvidia-ml-py",
                "segmentation-models-pytorch>=0.5.0"], check=False)

# smp pulls encoder weights from the HF hub on first build; fail loudly here
# rather than mid-sweep if the notebook has no internet.
import segmentation_models_pytorch as smp
print("segmentation-models-pytorch", smp.__version__)""".replace(
    "__REPO_URL__", REPO_URL)


CELL_SPLIT = """r = subprocess.run([sys.executable, "-m", "process_data.split",
    "--root", "data", "--out", "data/splits",
    "--block-size", "1000", "--buffer", "125"], capture_output=True, text=True)
print(r.stdout or r.stderr)"""


CELL_TRAIN = """RUNS = WORK / "runs"
RUNS.mkdir(exist_ok=True)
os.environ["RUNS_ROOT"] = str(RUNS)

# Architecture sweep (plan.md 4.2). The loss recipe is now fixed at the sweep-D
# winner (pos_weight 2.4, dice_weight 0.7 -> test IoU 0.5442); the open question
# is the encoder. Training a 14.8 M-param U-Net from scratch on 420 tiles was
# the binding constraint, not the loss - an ImageNet encoder should move it.
#
# U0 is the scratch control: same code path, no pretrained weights, so any gain
# from U1..U3 is unambiguously the encoder. stats_key auto-switches to ImageNet
# for the pretrained runs (model.registry.recommended_stats_key).
SWEEP = [
    ("U0_scratch",      ["--arch", "unet",       "--encoder", "scratch"],
     "verbatim 2023 U-Net, control"),
    ("U1_unet_rn34",    ["--arch", "unet",       "--encoder", "resnet34",
                         "--encoder-weights", "imagenet"],
     "U-Net + ResNet34/ImageNet - cheapest big win"),
    ("U2_unetpp_effb0", ["--arch", "unet++",     "--encoder", "efficientnet-b0",
                         "--encoder-weights", "imagenet"],
     "U-Net++ + EfficientNet-B0 - upgrade of the report's best model"),
    ("U3_dlv3p_effb2",  ["--arch", "deeplabv3+", "--encoder", "efficientnet-b2",
                         "--encoder-weights", "imagenet"],
     "DeepLabV3+ + EfficientNet-B2 - ASPP for multi-scale roofs"),
]

BASE = [sys.executable, "-u", "scripts/train_swiss.py",
        "--pos-weight", "2.4", "--dice-weight", "0.7",
        "--epochs", "80", "--batch-size", "16", "--lr", "3e-4",
        "--workers", "2", "--patience", "40", "--amp", "auto",
        "--gpu-util-target", "100", "--gpu-temp-limit", "0",
        "--gpu-mem-fraction", "0.95", "--checkpoint-every", "120",
        "--no-progress"]

t_all = time.time()
for name, extra, note in SWEEP:
    print("")
    print("=" * 72)
    print("RUN " + name + "  -  " + note)
    print("=" * 72, flush=True)
    cmd = BASE + ["--run-name", name] + extra
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    print(name + " exit=" + str(proc.returncode) +
          "  " + str(round((time.time() - t0) / 60, 1)) + " min", flush=True)

print("")
print("sweep total " + str(round((time.time() - t_all) / 60, 1)) + " min")"""


CELL_COLLECT = """summaries = []
for d in sorted(RUNS.iterdir()):
    f = d / "summary.json"
    if d.is_dir() and f.exists():
        summaries.append(json.loads(f.read_text()))

print("run              arch / encoder            ep  best_ep   val IoU   TEST IoU    P      R")
print("-" * 92)
best = None
for s in sorted(summaries, key=lambda x: -x["metrics"]["test"]["iou"]):
    m, c = s["metrics"]["test"], s["config"]
    enc = c.get("encoder", "scratch")
    desc = c.get("arch", "unet") + (" / " + enc if enc not in (None, "scratch") else " (scratch)")
    print(s["run"].ljust(16) + desc.ljust(26) +
          str(s["epochs_run"]).rjust(3) + str(s["best_epoch"]).rjust(8) +
          ("%.4f" % s["best_val_iou"]).rjust(10) +
          ("%.4f" % m["iou"]).rjust(11) +
          ("%.3f" % m["precision"]).rjust(7) +
          ("%.3f" % m["recall"]).rjust(7))
    if best is None:
        best = s

print("")
print("baseline to beat: test IoU 0.5442 (sweep D, scratch U-Net, 2026-08-22)")
if best:
    print("best here:        test IoU " + ("%.4f" % best["metrics"]["test"]["iou"]) +
          "  (" + best["run"] + ")")

# Ship only the winning weights plus every summary; the repo clone and the
# resume states must not go into the output or the download drags 169 MB back.
if best:
    shutil.copy2(RUNS / best["run"] / "best.pt", WORK / "best.pt")
    (WORK / "sweep.json").write_text(json.dumps(summaries, indent=2))
    for f in ("history.json", "train.log"):
        if (RUNS / best["run"] / f).exists():
            shutil.copy2(RUNS / best["run"] / f, WORK / f)
h = json.loads((RUNS / best["run"] / "history.json").read_text()) if best else None

for d in RUNS.iterdir():
    if d.is_dir():
        for junk in ("state.pt", "state.pt.bak", "state.pt.tmp", "last.pt", "best.pt"):
            (d / junk).unlink(missing_ok=True)
os.chdir(WORK)
shutil.rmtree(SRC, ignore_errors=True)
shutil.rmtree(RUNS, ignore_errors=True)
print("")
print("artifacts:", sorted(q.name for q in WORK.glob("*") if q.is_file()))"""


CELL_PLOT = """import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if h:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(h["train_loss"], label="train")
    ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title("loss")
    ax[1].plot(h["train_iou"], label="train")
    ax[1].plot(h["val_iou"], label="val")
    if h.get("best_epoch") is not None:
        ax[1].axvline(h["best_epoch"], ls="--", c="k", lw=0.8)
    ax[1].set_title("IoU")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(WORK / "curves.png", dpi=130)
    plt.show()"""


cells = [
    md("# Rooftop Solar - encoder sweep on Kaggle GPU\n\n"
       "Sweeps four segmentation architectures on the Swiss DOP25 set using the "
       "leakage-free geographic split (F-05) and the corrected metric harness "
       "(F-03 / F-08): a scratch U-Net control plus three ImageNet-pretrained "
       "encoders (`segmentation_models_pytorch`). Loss recipe is fixed at the "
       "sweep-D winner (pos_weight 2.4, dice_weight 0.7).\n\n"
       "Baseline to beat: **test IoU 0.5442**. Artifacts land in "
       "`/kaggle/working` and come back via `kaggle kernels output`."),
    md("## 0. GPU check, before importing torch"),
    code(CELL_GPU),
    md("## 1. Get the code\n\nThe repo carries the 169 MB Swiss dataset, so "
       "nothing needs uploading."),
    code(CELL_CLONE),
    md("## 2. Regenerate the geographic split\n\nDeterministic (seed 0), so "
       "this reproduces the local manifests exactly: 420 train / 58 val / "
       "74 test, with zero tiles adjacent across splits."),
    code(CELL_SPLIT),
    md("## 3. Sweep the encoders\n\nFour runs, loss recipe fixed. `U0_scratch` "
       "is the control - same code path, no pretrained weights - so any gain "
       "from `U1..U3` is unambiguously the encoder. The GPU governor is off "
       "here (duty-cycling a datacenter card only burns quota); AMP stays on "
       "`auto` so `utils.select_amp` picks fp16 on a T4 and skips it on a P100, "
       "after NaN-probing the real model."),
    code(CELL_TRAIN),
    md("## 4. Collect artifacts"),
    code(CELL_COLLECT),
    code(CELL_PLOT),
]

# Every code cell must still parse after the round-trip through splitlines().
problems = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    try:
        ast.parse("".join(c["source"]))
    except SyntaxError as exc:
        problems.append((i, str(exc)))

if problems:
    raise SystemExit(f"generated cells do not parse: {problems}")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with io.open("kaggle/kaggle_train.ipynb", "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)

print(f"notebook regenerated: {len(cells)} cells, all code cells parse")
