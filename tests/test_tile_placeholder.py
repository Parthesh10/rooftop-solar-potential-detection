"""A 200 OK carrying "Map data not yet available" is a failure, not a result.

Esri answers "no imagery at this zoom" with HTTP 200 and a grey placeholder
card, so ``raise_for_status()`` is happy and the tile counts as fetched. For a
tool whose entire output is an area estimate that is the worst possible failure
mode: the model is handed a grey rectangle, correctly predicts nothing, and the
user is told "0 buildings, 0 kWh" — indistinguishable from "we looked and there
are no roofs here".

Found 2026-09-09 while building rural training data: rural Bihar and rural
Maharashtra returned byte-identical mosaics (123 unique colours, mean 204.7,
std 5.4) where a real Delhi mosaic has 202,347 colours and std 49.
"""

import numpy as np
from PIL import Image, ImageDraw

from webapp.tiles import PLACEHOLDER_MAX_COLOURS, TILE_PX, _is_placeholder


def _esri_placeholder() -> np.ndarray:
    """A faithful stand-in: flat grey fill plus antialiased white text."""
    img = Image.new("RGB", (TILE_PX, TILE_PX), (205, 205, 205))
    d = ImageDraw.Draw(img)
    for y in (60, 190):
        d.text((30, y), "Map data not\nyet available", fill=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def _photography(seed: int = 0) -> np.ndarray:
    """Aerial photography always carries sensor noise, even over water."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(TILE_PX, TILE_PX, 3), dtype=np.uint8)
    return base


def test_detects_the_real_placeholder_shape():
    assert _is_placeholder(_esri_placeholder())


def test_does_not_flag_real_photography():
    assert not _is_placeholder(_photography())


def test_does_not_flag_low_contrast_photography():
    """Desert, open water and snow are legitimately flat but still photographs:
    they carry sensor noise across thousands of distinct values. Flagging them
    would silently discard real coverage."""
    rng = np.random.default_rng(1)
    calm_sea = (np.full((TILE_PX, TILE_PX, 3), 70, dtype=np.int16)
                + rng.integers(-6, 7, size=(TILE_PX, TILE_PX, 3)))
    arr = np.clip(calm_sea, 0, 255).astype(np.uint8)
    assert not _is_placeholder(arr)


def test_flags_a_uniform_fill():
    assert _is_placeholder(np.full((TILE_PX, TILE_PX, 3), 128, dtype=np.uint8))


def test_threshold_is_far_from_both_populations():
    """The measured gap is three orders of magnitude — 123 colours for the
    placeholder against 202,347 for real imagery — so the cut point is not
    finely balanced and should not need tuning."""
    assert _is_placeholder(_esri_placeholder())
    placeholder_colours = len(np.unique(
        _esri_placeholder().reshape(-1, 3), axis=0))
    photo_colours = len(np.unique(_photography().reshape(-1, 3), axis=0))
    assert placeholder_colours < PLACEHOLDER_MAX_COLOURS < photo_colours


def test_blank_tile_counts_as_placeholder():
    """The grey stand-in used for a genuinely failed fetch must also register,
    so a mosaic of failures cannot be mistaken for imagery downstream."""
    from webapp.tiles import _blank_tile
    assert _is_placeholder(_blank_tile())
