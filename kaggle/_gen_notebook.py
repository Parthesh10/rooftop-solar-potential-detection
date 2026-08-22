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
                "tifffile", "nvidia-ml-py"], check=False)""".replace(
    "__REPO_URL__", REPO_URL)


CELL_SPLIT = """r = subprocess.run([sys.executable, "-m", "process_data.split",
    "--root", "data", "--out", "data/splits",
    "--block-size", "1000", "--buffer", "125"], capture_output=True, text=True)
print(r.stdout or r.stderr)"""


CELL_TRAIN = """RUNS = WORK / "runs"
RUNS.mkdir(exist_ok=True)
os.environ["RUNS_ROOT"] = str(RUNS)

# Sweep. The first Kaggle run early-stopped at epoch 27 with its best at 12,
# so the cosine schedule never annealed and precision fell to 0.64 (from the
# 2023 model's 0.78) while recall rose - the auto-estimated pos_weight of ~5.9
# over-predicts. patience is raised to 40 so the schedule completes, and
# pos_weight is swept against the auto value.
SWEEP = [
    ("A_auto_pw",   [],                                  "patience 40, pos_weight auto (~5.9)"),
    ("B_pw2.4",     ["--pos-weight", "2.4"],             "sqrt of the imbalance"),
    ("C_pw1.0",     ["--pos-weight", "1.0"],             "no class weighting"),
    ("D_pw2.4_d0.7",["--pos-weight", "2.4", "--dice-weight", "0.7"], "more Dice"),
]

BASE = [sys.executable, "-u", "scripts/train_swiss.py",
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

print("run            pos_w  dice   ep  best_ep   val IoU   TEST IoU    P      R")
print("-" * 78)
best = None
for s in sorted(summaries, key=lambda x: -x["metrics"]["test"]["iou"]):
    m, c = s["metrics"]["test"], s["config"]
    pw = "auto" if c["pos_weight"] is None else str(c["pos_weight"])
    print(s["run"].ljust(14) + pw.ljust(7) + str(c["dice_weight"]).ljust(7) +
          str(s["epochs_run"]).rjust(3) + str(s["best_epoch"]).rjust(8) +
          ("%.4f" % s["best_val_iou"]).rjust(10) +
          ("%.4f" % m["iou"]).rjust(11) +
          ("%.3f" % m["precision"]).rjust(7) +
          ("%.3f" % m["recall"]).rjust(7))
    if best is None:
        best = s

print("")
print("baseline to beat: test IoU 0.4656 (Kaggle v5, 2026-08-22)")
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
    md("# Rooftop Solar - U-Net training on Kaggle GPU\n\n"
       "Trains the repaired U-Net on the Swiss DOP25 set using the "
       "leakage-free geographic split (F-05), with the corrected metric "
       "harness (F-03 / F-08).\n\n"
       "Artifacts land in `/kaggle/working` and come back via "
       "`kaggle kernels output`."),
    md("## 0. GPU check, before importing torch"),
    code(CELL_GPU),
    md("## 1. Get the code\n\nThe repo carries the 169 MB Swiss dataset, so "
       "nothing needs uploading."),
    code(CELL_CLONE),
    md("## 2. Regenerate the geographic split\n\nDeterministic (seed 0), so "
       "this reproduces the local manifests exactly: 420 train / 58 val / "
       "74 test, with zero tiles adjacent across splits."),
    code(CELL_SPLIT),
    md("## 3. Train\n\nThe GPU governor is off here - duty-cycling a "
       "datacenter card only burns quota. AMP stays on `auto` so "
       "`utils.select_amp` picks fp16 on a T4 (tensor cores) and skips it on "
       "a P100, after NaN-probing the real model."),
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
