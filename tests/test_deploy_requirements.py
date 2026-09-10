"""requirements-webapp.txt must describe an installable, tested environment.

Found 2026-09-10 while building the Dockerfile: the file pinned
``pyproj==3.8.1``, a version that does not exist on PyPI at all, and three
other packages at versions nobody had actually run the app against — the file
had only ever been hand-edited, never installed from cleanly. Anyone following
webapp/README.md's own first instruction, ``pip install -r
requirements-webapp.txt``, would have hit a hard failure before the app ever
started.

This does not re-run pip (slow, network-dependent, and the point is to catch
drift the moment it happens, not once a release is being cut). Instead it
checks that whatever this file pins is exactly what the environment running
the test suite already has installed and has therefore actually been exercised
against the whole app. That is a live assertion, not a snapshot: if a developer
bumps fastapi in .venv without updating this file, or hand-edits a pin without
installing it, this fails immediately and names the package.
"""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REQ_FILE = REPO / "requirements-webapp.txt"

PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?==([^\s#]+)")


def _parsed_pins() -> dict[str, str]:
    pins = {}
    for line in REQ_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = PIN_RE.match(line)
        if m:
            pins[m.group(1)] = m.group(3)
    return pins


def test_file_has_pins_to_check():
    """A guard against the test silently checking nothing — if parsing ever
    stops finding any pins, that is itself a bug worth failing loudly on."""
    assert len(_parsed_pins()) >= 5


def test_every_pin_matches_what_is_actually_installed():
    pins = _parsed_pins()
    mismatches = []
    for pkg, pinned in pins.items():
        try:
            installed = version(pkg)
        except PackageNotFoundError:
            mismatches.append(f"{pkg}: pinned {pinned}, but not installed at all")
            continue
        if installed != pinned:
            mismatches.append(f"{pkg}: pinned {pinned}, installed {installed}")
    assert not mismatches, (
        "requirements-webapp.txt is out of sync with the environment this "
        "test suite actually runs against (and the deploy image is built "
        "from): " + "; ".join(mismatches))
