# Joint training run B (collected 2026-09-06)

Output of `kaggle_joint/` as downloaded on 2026-09-06. Its `best.pt` is stored
as `results/joint_v3_effb0_20260906.pt`.

**This directory exists because `kaggle_joint_out/` holds a *different execution
of the same configuration*, and the pair is the project's only measurement of
repeat-run noise.** Keeping only the newer one would have destroyed that.

| | `kaggle_joint_out/` (A) | this directory (B) |
|---|---|---|
| `best_val_iou` (per-window) | 0.7174775 | 0.7174843 |
| epochs | 46 | 46 |
| epoch size / time | ~594 steps, 367 s | ~594 steps, 366 s |
| training loss per epoch | identical to 4 dp | identical to 4 dp |
| checkpoint | `results/joint_effb0_20260905.pt` | `results/joint_v3_effb0_20260906.pt` |

Scored, the two runs give:

| | A | B | spread |
|---|---|---|---|
| Inria pooled IoU @0.50 | 0.7661 | 0.7662 | 0.0001 |
| Bangalore footprint recall | 0.369 | 0.390 | 0.021 |
| silent fraction | 0.546 | 0.560 | 0.014 |

So **Inria reproduces to ~0.0001 and the Bangalore numbers move ~0.02 between
identical runs.** Any Indian result smaller than that, from a single run, is
noise. See `results/RESULTS.md` §"Finding 3".

The handoff notes described these as a 16-tile and a 103-tile run. The surviving
logs do not support that — both have the same epoch size, so both saw the same
amount of Indian data. The partial-mount incident was real (it is why
`kaggle_joint/_gen_notebook.py` now asserts `EXPECTED_EXTRA_TILES`), but no log
from that run is in this repo.
