"""Shared fixtures: a tiny synthetic paired dataset on disk."""

import numpy as np
import pytest
from PIL import Image

TILE = 64


def _make_tile(seed: int) -> tuple[Image.Image, Image.Image]:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(TILE, TILE, 3), dtype=np.uint8)
    mask = np.zeros((TILE, TILE), dtype=np.uint8)
    y, x = rng.integers(0, TILE // 2, size=2)
    mask[y : y + TILE // 4, x : x + TILE // 4] = 255
    return Image.fromarray(img), Image.fromarray(mask, mode="L")


@pytest.fixture
def paired_dataset(tmp_path):
    """``tmp_path`` containing images/ and labels/ with 6 matched pairs."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    names = []
    for i in range(6):
        # Deliberately non-monotonic stems, so a positional pairing bug shows up.
        stem = f"DOP25_LV03_1301_11_2015_1_15_{497500 + i * 62.5}_{119000 + i * 62.5}"
        img, mask = _make_tile(i)
        img.save(img_dir / f"{stem}.png")
        mask.save(lbl_dir / f"{stem}_label.png")
        names.append(stem)
    return {"root": tmp_path, "images": img_dir, "labels": lbl_dir, "names": names}
