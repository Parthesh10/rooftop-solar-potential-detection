"""Regenerate kaggle_joint_train.ipynb.

Same discipline as kaggle/_gen_notebook.py: every code cell is parsed with ``ast``
before the notebook is written, because a stray ``\\n`` in a cell string gets
turned into a real newline by ``splitlines(keepends=True)`` and silently cuts a
statement in half.

This notebook trains ONE model on Inria **plus** the hand-selected Indian tiles
relabelled from OpenStreetMap, to test whether a single checkpoint can be good
in both places.

Why this run is coherent and the obvious version of it was not: the hand-drawn
Indian labels merged adjacent buildings into cluster envelopes, and mixing those
with Inria's footprints would teach the model two different answers to "where
does a roof end". ``scripts/build_osm_labels.py`` relabels the same tiles from
OpenStreetMap — one polygon per building, human-drawn — so both halves of this
training set now mean the same thing. Pixels that are inside a hand envelope but
outside any OSM building are marked *ignore* rather than background, because
they are genuinely unknown (an alley, or a roof nobody mapped) and calling them
background is what teaches a model to go silent on real rooftops.

There is no Inria-only control in the sweep on purpose: the shipped model
(pooled val IoU 0.7712) already is that control.
"""

import ast
import io
import json

REPO_URL = "https://github.com/Parthesh10/rooftop-solar-potential-detection.git"
INRIA_SLUG = "inria-aerial-image-labeling-dataset"
INDIAN_SLUG = "partheshgupta/rooftop-solar-indian-osm-tiles"


def md(t):
    return {"id": f"m{abs(hash(t)) % 99999}", "cell_type": "markdown",
            "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"id": f"c{abs(hash(t)) % 99999}", "cell_type": "code",
            "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


# GPU check — identical policy to the Swiss notebook: nvidia-smi, then a real
# matmul in a fresh interpreter, then a cu118 reinstall if the stock torch
# cannot drive the card (P100 = sm_60 vs a torch built for sm_70+).
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
    raise SystemExit("No GPU. Push with: kaggle kernels push -p kaggle_joint --accelerator gpuT4x2")

PROBE = (
    "import torch;"
    "p=torch.cuda.get_device_properties(0);"
    "cap='sm_'+str(p.major)+str(p.minor);"
    "t=torch.randn(256,256,device='cuda');"
    "ok=bool(torch.isfinite(t@t).all());"
    "print(torch.__version__, cap, cap in torch.cuda.get_arch_list(), ok)"
)

def probe():
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
    return r.stdout.strip(), r.returncode, r.stderr.strip()[-400:]

out, rc, err = probe()
print("probe:", out or err)
if rc != 0 or "True True" not in out:
    print("this torch cannot drive this GPU - installing a cu118 build")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "torch==2.4.1+cu118", "torchvision==0.19.1+cu118",
         "--index-url", "https://download.pytorch.org/whl/cu118"],
        capture_output=True, text=True)
    print(r.stdout[-1200:] if r.returncode == 0 else r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("cu118 torch install failed")
    out, rc, err = probe()
    print("probe after reinstall:", out or err)
if rc != 0 or "True True" not in out:
    raise SystemExit("GPU still unusable (" + (out or err) + "). Try --accelerator gpuT4x2")
print("GPU verified by real matmul in a fresh interpreter")"""


CELL_SETUP = """REPO = "__REPO_URL__"
WORK = Path("/kaggle/working")
SRC = WORK / "repo"

if SRC.exists():
    shutil.rmtree(SRC)
subprocess.run(["git", "clone", "--depth", "1", REPO, str(SRC)], check=True)
os.chdir(SRC)
sys.path.insert(0, str(SRC))
print("cloned at", sh(["git", "rev-parse", "--short", "HEAD"]))

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "tifffile", "nvidia-ml-py",
                "segmentation-models-pytorch>=0.5.0"], check=False)
import segmentation_models_pytorch as smp
print("segmentation-models-pytorch", smp.__version__)

# Find the mounted Inria dataset. dataset_sources mounts it under /kaggle/input;
# don't hardcode the exact nesting - just locate AerialImageDataset.
INRIA = None
for base in Path("/kaggle/input").glob("**/AerialImageDataset"):
    if (base / "train" / "images").is_dir():
        INRIA = base
        break
if INRIA is None:
    raise SystemExit(
        "Inria not mounted. Add it in the notebook's Data panel or set "
        "dataset_sources=['sagar100rathod/__INRIA_SLUG__'] in kernel-metadata.json")
n_train = len(list((INRIA / "train" / "images").glob("*.tif")))
print("Inria at", INRIA, "-", n_train, "train tiles")

# The Indian tiles: images/ + labels/ pairs, uploaded as a Kaggle dataset
# because data/ is gitignored in the repo.
# How many tiles the dataset is supposed to contain. The first joint run
# trained on 16 of them instead of all of them and said nothing: the kernel was
# pushed ~2 minutes after the dataset was created, so Kaggle was still
# processing the upload and mounted a partial copy. The India half of the run
# was therefore ~5% of an epoch instead of ~22%, and the only visible symptom
# was a muted result. Assert the count rather than trust the mount.
EXPECTED_EXTRA_TILES = 103

EXTRA = None
best_n = 0
for base in sorted(Path("/kaggle/input").glob("**/images")):
    labels = base.parent / "labels"
    if not labels.is_dir() or "AerialImage" in str(base):
        continue
    n = len(list(base.glob("*.png")))
    # Take the richest match, not the first: glob order is not meaningful and a
    # partial or stray directory must not win by being earlier.
    if n > best_n:
        EXTRA, best_n = base.parent, n
if EXTRA is None:
    raise SystemExit(
        "Indian tiles not mounted. Add partheshgupta/rooftop-solar-indian-osm-tiles in the Data panel or "
        "in dataset_sources.")

n_lbl = len(list((EXTRA / "labels").glob("*_label.png")))
print("Indian tiles at", EXTRA, "-", best_n, "images,", n_lbl, "labels")
if best_n < EXPECTED_EXTRA_TILES or n_lbl < EXPECTED_EXTRA_TILES:
    raise SystemExit(
        "Only " + str(best_n) + " images / " + str(n_lbl) + " labels mounted, "
        "expected " + str(EXPECTED_EXTRA_TILES) + ". The dataset is probably "
        "still processing - wait for it to finish on the dataset page, then "
        "re-push. Training on a partial mount silently under-weights the new "
        "region, which is exactly the failure this check exists to catch.")""".replace(
    "__REPO_URL__", REPO_URL).replace("__INRIA_SLUG__", INRIA_SLUG)


CELL_TRAIN = """RUNS = WORK / "runs"
RUNS.mkdir(exist_ok=True)
os.environ["RUNS_ROOT"] = str(RUNS)

# One run, not a sweep. The Inria-only control already exists - it is the
# shipped model, pooled val IoU 0.7712 - so spending half the session
# reproducing it would buy nothing.
#
# --extra-repeat 20: Inria contributes 155 tiles x 48 windows = ~7.4k windows an
# epoch. 103 Indian tiles unrepeated would be 1.4% of the gradient and would
# change nothing measurable. Repeating them 20x makes them ~22% of each epoch.
# Repetition is not the same as having more data and it will overfit those tiles
# if pushed much further; 20 is the point where the region is represented
# without dominating.
SWEEP = [
    ("J1_joint_effb0", ["--arch", "unet++", "--encoder", "efficientnet-b0",
                        "--samples-per-tile", "48", "--epochs", "50",
                        "--extra-data-dir", str(EXTRA),
                        "--extra-ignore-value", "128",
                        "--extra-repeat", "20"],
     "U-Net++ / EfficientNet-B0 on Inria + Indian OSM tiles"),
]

BASE = [sys.executable, "-u", "scripts/train_inria.py",
        "--inria-root", str(INRIA),
        "--encoder-weights", "imagenet",
        "--window", "512", "--val-stride", "512",
        "--pos-weight", "2.4", "--dice-weight", "0.6",
        "--batch-size", "16", "--lr", "3e-4",
        "--workers", "2", "--patience", "12", "--amp", "auto",
        "--data-parallel",
        "--gpu-util-target", "100", "--gpu-temp-limit", "0",
        "--gpu-mem-fraction", "0.95", "--checkpoint-every", "300",
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
print("total " + str(round((time.time() - t_all) / 60, 1)) + " min")"""


CELL_COLLECT = """summaries = []
for d in sorted(RUNS.iterdir()):
    f = d / "summary.json"
    if d.is_dir() and f.exists():
        summaries.append(json.loads(f.read_text()))

print("run              arch / encoder             ep  best_ep   VAL IoU     P      R")
print("-" * 78)
best = None
for s in sorted(summaries, key=lambda x: -x["best_val_iou"]):
    c = s["config"]
    m = s["metrics"].get("val", {})
    desc = c.get("arch", "unet") + " / " + str(c.get("encoder"))
    print(s["run"].ljust(16) + desc.ljust(27) +
          str(s["epochs_run"]).rjust(3) + str(s["best_epoch"]).rjust(8) +
          ("%.4f" % s["best_val_iou"]).rjust(10) +
          ("%.3f" % m.get("precision", float("nan"))).rjust(7) +
          ("%.3f" % m.get("recall", float("nan"))).rjust(7))
    if best is None:
        best = s

print("")
print("published Inria building-seg work: ~0.78-0.82 IoU;  plan.md target: >= 0.72")
if best:
    print("best here: VAL IoU " + ("%.4f" % best["best_val_iou"]) + "  (" + best["run"] + ")")

# Ship the winning weights + every summary; drop the repo clone and resume states.
if best:
    shutil.copy2(RUNS / best["run"] / "best.pt", WORK / "best_inria.pt")
    (WORK / "sweep_inria.json").write_text(json.dumps(summaries, indent=2))
    for f in ("history.json", "train.log", "metadata.json"):
        if (RUNS / best["run"] / f).exists():
            shutil.copy2(RUNS / best["run"] / f, WORK / (best["run"] + "_" + f))
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
    ax[1].set_title("IoU  (Inria official val)")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(WORK / "curves_inria.png", dpi=130)
    plt.show()"""


cells = [
    md("# Rooftop Solar - Inria training on Kaggle GPU\n\n"
       "Trains a **general** rooftop-detection model on the full Inria Aerial "
       "Image Labeling dataset (180 tiles, 5000x5000 @ 0.3 m/px, five cities), "
       "mounted from the public Kaggle copy - no upload. Labels are building "
       "footprints; the usable-for-PV fraction is a downstream packing factor.\n\n"
       "Split: Inria's official protocol (tiles 1-5 -> val, 6-36 -> train), so "
       "val IoU is comparable to published work (~0.78-0.82). Target: >= 0.72."),
    md("## 0. GPU check, before importing torch"),
    code(CELL_GPU),
    md("## 1. Code + data\n\nThe repo is cloned for the code; Inria is mounted "
       "via `dataset_sources` and located under `/kaggle/input`."),
    code(CELL_SETUP),
    md("## 2. Train\n\nWindowed 512x512 reads straight from the GeoTIFFs "
       "(`process_data.inria.InriaWindowDataset`) - no pre-cutting to disk. "
       "AMP `auto` -> fp16 on a T4. Checkpoints every 5 min and every epoch, so "
       "`scripts/train_inria.py --resume` continues in a second session if the "
       "12 h cap is hit."),
    code(CELL_TRAIN),
    md("## 3. Collect artifacts"),
    code(CELL_COLLECT),
    code(CELL_PLOT),
]

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

with io.open("kaggle_joint/kaggle_joint_train.ipynb", "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)

print(f"notebook regenerated: {len(cells)} cells, all code cells parse")
