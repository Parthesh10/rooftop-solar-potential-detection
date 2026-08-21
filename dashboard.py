"""Localhost training dashboard — live metrics and run control in a browser.

    python dashboard.py                 # http://127.0.0.1:8420
    python dashboard.py --port 9000 --open

Shows, for every run under ``runs/``:

* live loss / IoU curves and within-epoch progress (the training loop writes a
  ``status.json`` heartbeat about once a second);
* GPU utilisation, VRAM, temperature, power draw, CPU and RAM;
* the run log, tailed;
* Pause / Resume / Stop buttons, which drive the same ``PAUSE`` / ``STOP``
  control files that ``runstate.RunControl`` watches — so the browser and the
  terminal are interchangeable, and closing the browser cannot orphan anything;
* a Start form that launches ``scripts/train_swiss.py`` as a detached process.

Deliberately built on ``http.server`` and inline HTML/CSS/JS: no FastAPI, no
uvicorn, no CDN. It is a local control panel for one user, and adding a web
framework to a training repo for that is not a good trade. It binds to
127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import RUNS_ROOT  # noqa: E402

try:
    from sysmon import SysMon
except Exception:  # sysmon pulls torch; the dashboard should still run without it
    SysMon = None

_MON = None
_MON_LOCK = threading.Lock()


def monitor():
    """Lazily create one shared SysMon; NVML handles are not free to re-open."""
    global _MON
    with _MON_LOCK:
        if _MON is None and SysMon is not None:
            try:
                _MON = SysMon(enabled=True)
            except Exception:
                pass
    return _MON


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_dirs() -> list[Path]:
    if not RUNS_ROOT.is_dir():
        return []
    return sorted(
        (d for d in RUNS_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def run_summary(d: Path) -> dict:
    status = _read_json(d / "status.json") or {}
    hist = _read_json(d / "history.json") or status.get("history") or {}
    updated = status.get("updated", 0)
    state = status.get("state", "unknown")

    # A heartbeat older than 15 s means the process is gone, whatever it last
    # claimed. Without this a killed run shows as "running" forever.
    stale = (time.time() - updated) > 15 if updated else True
    if stale and state in ("running", "paused"):
        state = "dead"

    return {
        "name": d.name,
        "state": state,
        "paused": (d / "PAUSE").exists(),
        "stopping": (d / "STOP").exists(),
        "epoch": status.get("epoch"),
        "num_epochs": status.get("num_epochs"),
        "step": status.get("step"),
        "train_batches": status.get("train_batches"),
        "running_loss": status.get("running_loss"),
        "running_iou": status.get("running_iou"),
        "images_per_sec": status.get("images_per_sec"),
        "best_val_iou": status.get("best_val_iou") or hist.get("best_val_iou"),
        "best_epoch": status.get("best_epoch") if status.get("best_epoch") is not None
                      else hist.get("best_epoch"),
        "device": status.get("device"),
        "amp": status.get("amp"),
        "amp_dtype": status.get("amp_dtype"),
        "batch_size": status.get("batch_size"),
        "updated": updated,
        "age": (time.time() - updated) if updated else None,
        "history": hist,
        "has_best": (d / "best.pt").exists(),
        "resumable": (d / "state.pt").exists(),
        "mtime": d.stat().st_mtime,
    }


def tail(path: Path, n: int = 200) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def launch_training(params: dict) -> dict:
    """Spawn scripts/train_swiss.py detached, logging into the repo."""
    script = REPO / "scripts" / "train_swiss.py"
    if not script.exists():
        return {"ok": False, "error": f"missing {script}"}

    py = REPO / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)

    cmd = [str(py), "-u", str(script), "--no-progress"]
    for flag, key in (
        ("--epochs", "epochs"), ("--batch-size", "batch_size"), ("--lr", "lr"),
        ("--workers", "workers"), ("--patience", "patience"), ("--amp", "amp"),
    ):
        v = params.get(key)
        if v not in (None, ""):
            cmd += [flag, str(v)]
    if params.get("resume"):
        cmd += ["--resume"]

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    logdir = RUNS_ROOT / "_dashboard"
    logdir.mkdir(exist_ok=True)
    logfile = logdir / f"launch_{datetime.now():%Y%m%d_%H%M%S}.log"

    # CREATE_NEW_PROCESS_GROUP so Ctrl+C in the dashboard's terminal does not
    # kill training, and CREATE_NO_WINDOW to keep it headless.
    #
    # Do NOT add DETACHED_PROCESS: it breaks the DataLoader's multiprocessing
    # workers on Windows. Measured failure — "_pickle.UnpicklingError: pickle
    # data was truncated" from spawn_main, plus the Intel Fortran runtime in
    # MKL aborting with "forrtl: error (200): program aborting due to
    # window-CLOSE event". The run dies before writing a single heartbeat.
    flags = 0
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    try:
        fh = open(logfile, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=flags)
        return {"ok": True, "pid": p.pid, "log": str(logfile), "cmd": " ".join(cmd)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "rsolar-dashboard"

    def log_message(self, *a):  # keep the console clean for training output
        pass

    # ---------- helpers ---------- #
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _safe_run(self, name: str) -> Path | None:
        """Resolve a run name, rejecting anything outside RUNS_ROOT."""
        if not name:
            return None
        d = (RUNS_ROOT / name).resolve()
        try:
            d.relative_to(RUNS_ROOT.resolve())
        except ValueError:
            return None  # path traversal attempt
        return d if d.is_dir() else None

    # ---------- routes ---------- #
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

        if u.path == "/api/runs":
            return self._json({"runs": [run_summary(d) for d in run_dirs()]})

        if u.path == "/api/run":
            d = self._safe_run((q.get("name") or [""])[0])
            if d is None:
                return self._json({"error": "unknown run"}, 404)
            out = run_summary(d)
            out["log"] = tail(d / "train.log", int((q.get("lines") or [200])[0]))
            out["metadata"] = _read_json(d / "metadata.json")
            return self._json(out)

        if u.path == "/api/sys":
            m = monitor()
            return self._json(m.sample().to_dict() if m else {})

        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}

        if u.path == "/api/start":
            return self._json(launch_training(body))

        if u.path in ("/api/pause", "/api/resume", "/api/stop"):
            d = self._safe_run(body.get("name", ""))
            if d is None:
                return self._json({"error": "unknown run"}, 404)
            action = u.path.rsplit("/", 1)[-1]
            try:
                if action == "pause":
                    (d / "PAUSE").touch()
                elif action == "resume":
                    (d / "PAUSE").unlink(missing_ok=True)
                else:
                    (d / "STOP").touch()
                return self._json({"ok": True, "action": action, "run": d.name})
            except OSError as exc:
                return self._json({"ok": False, "error": str(exc)}, 500)

        return self._json({"error": "not found"}, 404)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rooftop Solar — training dashboard</title>
<style>
:root{
  --bg:#0e1116; --panel:#161b22; --panel2:#1c2230; --line:#2a3140;
  --fg:#e6edf3; --dim:#8b949e; --accent:#4a9eff; --ok:#3fb950; --warn:#d29922;
  --bad:#f85149; --pause:#a371f7;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.5 ui-monospace,"Cascadia Code",Consolas,monospace}
header{display:flex;align-items:center;gap:16px;padding:12px 20px;
  background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap;
  position:sticky;top:0;z-index:10}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
.wrap{padding:20px;display:grid;gap:16px;grid-template-columns:320px 1fr}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--dim);
  margin:0 0 10px}
.runs{display:flex;flex-direction:column;gap:6px;max-height:300px;overflow:auto}
.run{padding:8px 10px;border-radius:6px;background:var(--panel2);cursor:pointer;
  border:1px solid transparent;display:flex;justify-content:space-between;gap:8px}
.run:hover{border-color:var(--line)}
.run.sel{border-color:var(--accent)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px}
.s-running{background:var(--ok)} .s-paused{background:var(--pause)}
.s-finished{background:var(--accent)} .s-stopped{background:var(--warn)}
.s-dead{background:var(--bad)} .s-unknown{background:var(--dim)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.stat{background:var(--panel2);border-radius:6px;padding:10px}
.stat .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.stat .v{font-size:20px;font-weight:600;margin-top:2px}
.stat .v.sm{font-size:14px;font-weight:500}
button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);
  border-radius:6px;padding:7px 14px;cursor:pointer;font:inherit;font-size:13px}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.35;cursor:not-allowed}
button.p{border-color:var(--pause)} button.s{border-color:var(--bad)}
button.g{border-color:var(--ok)}
.bar{height:7px;background:var(--panel2);border-radius:4px;overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;background:var(--accent);transition:width .3s}
pre{background:#0a0d12;border:1px solid var(--line);border-radius:6px;padding:10px;
  max-height:280px;overflow:auto;font-size:12px;margin:0;white-space:pre-wrap}
label{display:block;color:var(--dim);font-size:11px;margin-bottom:3px;
  text-transform:uppercase;letter-spacing:.5px}
input,select{width:100%;background:var(--panel2);border:1px solid var(--line);
  color:var(--fg);border-radius:5px;padding:6px 8px;font:inherit;font-size:13px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:8px}
.meter{height:6px;background:var(--panel2);border-radius:3px;overflow:hidden;margin-top:4px}
.meter>i{display:block;height:100%;transition:width .4s}
svg{width:100%;height:200px;display:block}
.legend{display:flex;gap:14px;font-size:11px;color:var(--dim);margin-top:6px}
.legend b{font-weight:500}
.muted{color:var(--dim)}
.pill{font-size:11px;padding:2px 8px;border-radius:10px;background:var(--panel2);
  color:var(--dim)}
</style></head><body>

<header>
  <h1>◉ Rooftop Solar — training</h1>
  <span id="conn" class="pill">connecting…</span>
  <span style="flex:1"></span>
  <span id="gpuline" class="muted"></span>
</header>

<div class="wrap">
  <div style="display:flex;flex-direction:column;gap:16px">
    <div class="card">
      <h2>Runs</h2>
      <div id="runs" class="runs"></div>
    </div>

    <div class="card">
      <h2>Control</h2>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button id="bPause" class="p">⏸ Pause</button>
        <button id="bResume" class="g">▶ Resume</button>
        <button id="bStop" class="s">⏹ Stop</button>
      </div>
      <p class="muted" style="font-size:11px;margin:10px 0 0">
        These write PAUSE / STOP files the trainer polls, so they work whether it
        was started here or from a terminal. Stop saves a resumable checkpoint.
      </p>
    </div>

    <div class="card">
      <h2>Start a run</h2>
      <div class="row">
        <div><label>epochs</label><input id="f_epochs" value="80"></div>
        <div><label>batch</label><input id="f_batch_size" value="8"></div>
        <div><label>lr</label><input id="f_lr" value="3e-4"></div>
        <div><label>workers</label><input id="f_workers" value="2"></div>
        <div><label>patience</label><input id="f_patience" value="15"></div>
        <div><label>amp</label><select id="f_amp">
          <option>auto</option><option>off</option><option>bf16</option><option>fp16</option>
        </select></div>
      </div>
      <div style="display:flex;gap:8px;margin-top:10px">
        <button id="bStart" class="g">Start</button>
        <button id="bResumeRun">Resume newest</button>
      </div>
      <div id="startmsg" class="muted" style="font-size:11px;margin-top:8px"></div>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:16px">
    <div class="card">
      <h2>Progress — <span id="runName" class="muted">no run selected</span></h2>
      <div class="grid" id="stats"></div>
      <div style="margin-top:12px">
        <div class="muted" style="font-size:11px" id="epochLabel">—</div>
        <div class="bar"><i id="epochBar" style="width:0%"></i></div>
      </div>
      <div style="margin-top:8px">
        <div class="muted" style="font-size:11px" id="stepLabel">—</div>
        <div class="bar"><i id="stepBar" style="width:0%"></i></div>
      </div>
    </div>

    <div class="card">
      <h2>Curves</h2>
      <svg id="chart" viewBox="0 0 800 200" preserveAspectRatio="none"></svg>
      <div class="legend">
        <span><b style="color:#4a9eff">━</b> train IoU</span>
        <span><b style="color:#3fb950">━</b> val IoU</span>
        <span><b style="color:#d29922">━</b> train loss</span>
        <span><b style="color:#f85149">━</b> val loss</span>
      </div>
    </div>

    <div class="card">
      <h2>Resources</h2>
      <div class="grid" id="sys"></div>
    </div>

    <div class="card">
      <h2>Log</h2>
      <pre id="log">—</pre>
    </div>
  </div>
</div>

<script>
let sel = null, runsCache = [];
const $ = id => document.getElementById(id);
const f = (v, d=4) => (v===null||v===undefined||Number.isNaN(v)) ? "—" : (+v).toFixed(d);
const gb = b => b===undefined||b===null ? "—" : (b/1073741824).toFixed(1);

async function jget(u){ const r = await fetch(u); return r.json(); }
async function jpost(u, body){
  const r = await fetch(u, {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(body||{})});
  return r.json();
}

function stat(k, v, cls){ return `<div class="stat"><div class="k">${k}</div>
  <div class="v ${cls||''}">${v}</div></div>`; }

function meter(k, v, pct, color){
  return `<div class="stat"><div class="k">${k}</div><div class="v sm">${v}</div>
    <div class="meter"><i style="width:${Math.max(0,Math.min(100,pct||0))}%;
      background:${color}"></i></div></div>`;
}

async function refreshRuns(){
  const d = await jget("/api/runs");
  runsCache = d.runs || [];
  if (!sel && runsCache.length) sel = runsCache[0].name;
  $("runs").innerHTML = runsCache.map(r => `
    <div class="run ${r.name===sel?'sel':''}" data-n="${r.name}">
      <span><span class="dot s-${r.state}"></span>${r.name}</span>
      <span class="muted">${r.epoch!==null&&r.epoch!==undefined ? "ep "+r.epoch : ""}</span>
    </div>`).join("") || '<span class="muted">no runs yet</span>';
  document.querySelectorAll(".run").forEach(el =>
    el.onclick = () => { sel = el.dataset.n; refreshRun(); refreshRuns(); });
}

function drawChart(h){
  const svg = $("chart");
  const ti = h.train_iou||[], vi = h.val_iou||[], tl = h.train_loss||[], vl = h.val_loss||[];
  const n = Math.max(ti.length, vi.length, tl.length, vl.length);
  if (n < 2){ svg.innerHTML = '<text x="400" y="100" fill="#8b949e" font-size="12" '+
    'text-anchor="middle">waiting for a second epoch…</text>'; return; }
  const W=800, H=200, P=6;
  const maxLoss = Math.max(...tl.filter(Number.isFinite), ...vl.filter(Number.isFinite), 1e-6);
  const line = (arr, color, norm) => {
    const pts = arr.map((v,i) => {
      if (!Number.isFinite(v)) return null;
      const x = P + i*(W-2*P)/Math.max(n-1,1);
      const y = H-P - (norm(v))*(H-2*P);
      return x.toFixed(1)+","+y.toFixed(1);
    }).filter(Boolean).join(" ");
    return pts ? `<polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}"/>` : "";
  };
  let grid = "";
  for (let g=0; g<=4; g++){ const y = P + g*(H-2*P)/4;
    grid += `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="#2a3140" stroke-width="1"/>`; }
  svg.innerHTML = grid
    + line(tl, "#d29922", v => v/maxLoss)
    + line(vl, "#f85149", v => v/maxLoss)
    + line(ti, "#4a9eff", v => v)
    + line(vi, "#3fb950", v => v);
}

async function refreshRun(){
  if (!sel){ return; }
  const r = await jget("/api/run?name="+encodeURIComponent(sel));
  if (r.error) return;
  $("runName").textContent = r.name + " · " + r.state;

  $("stats").innerHTML =
      stat("best val IoU", f(r.best_val_iou))
    + stat("best epoch", r.best_epoch ?? "—")
    + stat("train IoU", f(r.running_iou))
    + stat("loss", f(r.running_loss))
    + stat("img/s", f(r.images_per_sec,1))
    + stat("batch", r.batch_size ?? "—", "sm")
    + stat("precision", (r.amp ? r.amp_dtype.replace("torch.","") : "fp32"), "sm")
    + stat("resumable", r.resumable ? "yes" : "no", "sm");

  const ep = r.epoch ?? 0, ne = r.num_epochs ?? 1;
  $("epochLabel").textContent = `epoch ${ep} / ${ne-1}`;
  $("epochBar").style.width = (100*(ep+1)/Math.max(ne,1))+"%";
  const st = r.step ?? 0, nb = r.train_batches ?? 1;
  $("stepLabel").textContent = `step ${st} / ${nb}`;
  $("stepBar").style.width = (100*st/Math.max(nb,1))+"%";

  drawChart(r.history||{});
  $("log").textContent = r.log || "—";

  const live = r.state === "running" || r.state === "paused";
  $("bPause").disabled = !live || r.paused;
  $("bResume").disabled = !live || !r.paused;
  $("bStop").disabled = !live;
}

async function refreshSys(){
  const s = await jget("/api/sys");
  const vramPct = s.gpu_mem_total ? 100*s.gpu_mem_used/s.gpu_mem_total : 0;
  const ramPct  = s.ram_total ? 100*s.ram_used/s.ram_total : 0;
  $("sys").innerHTML =
      meter("GPU", (s.gpu_util??"—")+"%", s.gpu_util, "#4a9eff")
    + meter("VRAM", gb(s.gpu_mem_used)+" / "+gb(s.gpu_mem_total)+" GB", vramPct, "#a371f7")
    + meter("Temp", (s.gpu_temp??"—")+" °C", (s.gpu_temp/95)*100, "#d29922")
    + meter("Power", (s.gpu_power?s.gpu_power.toFixed(0):"—")+" / "
        +(s.gpu_power_cap?s.gpu_power_cap.toFixed(0):"—")+" W",
        s.gpu_power_cap ? 100*s.gpu_power/s.gpu_power_cap : 0, "#3fb950")
    + meter("CPU", (s.cpu_percent??"—")+"%", s.cpu_percent, "#4a9eff")
    + meter("RAM", gb(s.ram_used)+" / "+gb(s.ram_total)+" GB", ramPct, "#8b949e");
  $("gpuline").textContent = s.gpu_util!==undefined
    ? `gpu ${s.gpu_util}% · ${gb(s.gpu_mem_used)}/${gb(s.gpu_mem_total)}GB · ${s.gpu_temp}°C`
    : "";
  $("conn").textContent = "live";
}

$("bPause").onclick  = () => jpost("/api/pause",  {name:sel}).then(refreshRun);
$("bResume").onclick = () => jpost("/api/resume", {name:sel}).then(refreshRun);
$("bStop").onclick   = () => jpost("/api/stop",   {name:sel}).then(refreshRun);

async function start(resume){
  const body = {resume: !!resume};
  ["epochs","batch_size","lr","workers","patience","amp"].forEach(k =>
    body[k] = $("f_"+k).value);
  $("startmsg").textContent = "launching…";
  const r = await jpost("/api/start", body);
  $("startmsg").textContent = r.ok
    ? `started pid ${r.pid} — a new run appears within a few seconds`
    : `failed: ${r.error}`;
  setTimeout(refreshRuns, 3000);
}
$("bStart").onclick = () => start(false);
$("bResumeRun").onclick = () => start(true);

async function tick(){
  try { await refreshSys(); await refreshRun(); }
  catch(e){ $("conn").textContent = "disconnected"; }
}
refreshRuns(); tick();
setInterval(tick, 1000);
setInterval(refreshRuns, 4000);
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Localhost training dashboard")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open a browser tab")
    args = ap.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"dashboard: {url}")
    print(f"runs dir : {RUNS_ROOT}")
    print("Ctrl+C to stop the dashboard (this does NOT stop training)")
    if args.open:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
