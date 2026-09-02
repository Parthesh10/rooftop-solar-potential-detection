"""Run the app: ``python -m webapp``.

Kept deliberately dependency-light at import time so ``--help`` works even if
the model or onnxruntime is missing — the server should still start and tell the
user what to fix, rather than failing to boot.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m webapp",
        description="Rooftop solar potential — local web app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-reload on edit")
    ap.add_argument("--no-open", action="store_true",
                    help="do not open a browser window")
    args = ap.parse_args()

    from webapp.config import MODELS_DIR

    if not any(MODELS_DIR.glob("*.onnx")) and not any(MODELS_DIR.glob("*.json")):
        print("!" * 72)
        print("No model found in webapp/models/. Export one first:")
        print("    python scripts/export_onnx.py")
        print("The UI will start but analysis will return 503.")
        print("!" * 72)

    url = f"http://{args.host}:{args.port}"
    print(f"\n  Rooftop Solar Potential  ->  {url}\n")

    if not args.no_open and not args.reload:
        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    uvicorn.run("webapp.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
