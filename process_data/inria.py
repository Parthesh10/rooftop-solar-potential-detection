"""Inria Aerial Image Labeling dataset with on-the-fly windowed reads.

Implements plan.md §A4. The 2023 approach was to pre-cut 51,840 PNG crops to
disk; this reads random 512x512 windows straight out of the 5000x5000 GeoTIFFs
instead, so it costs no extra disk and every epoch sees different crops.

Why memory-mapping works here: the tiles are ~72 MB each and
5000 x 5000 x 3 bytes = 75 MB, so they are essentially uncompressed. ``tifffile``
can memmap them and slice a window without decoding the whole image.

Two things about this dataset that matter more than the code:

1. **The labels are building footprints, not available rooftop area.** A model
   trained here predicts roof extent; converting that to installable area needs
   a packing factor. Do not silently call the output "available area".
2. **The `test/` cities ship without ground truth.** Score on a held-out slice
   of `train/` — see ``split.inria_official_split``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image

from config import INRIA_WINDOW, norm_stats
from process_data.data_loader import D4_OPS

__all__ = ["InriaWindowDataset", "open_tile"]

Image.MAX_IMAGE_PIXELS = None  # 25 MP tiles trip PIL's decompression-bomb guard


def open_tile(path: Path):
    """Return an array-like supporting ``arr[y0:y1, x0:x1]`` without full decode.

    Prefers ``tifffile.memmap``; falls back to reading the whole tile with PIL,
    which works but costs 75 MB and a full decode per access.
    """
    try:
        import tifffile

        return tifffile.memmap(str(path), mode="r")
    except Exception:
        return np.asarray(Image.open(path))


class InriaWindowDataset(data.Dataset):
    """Random (train) or deterministic (eval) windows over Inria tiles.

    Args:
        image_files: list of ``<city><n>.tif`` paths under ``AerialImageDataset/train/images``.
        gt_dir: matching ``gt/`` directory. ``None`` for unlabelled tiles.
        window: side length of each crop. Must be a multiple of 32.
        samples_per_tile: how many windows one epoch draws from each tile.
            Training length is ``len(image_files) * samples_per_tile``. Lower
            this to make an epoch tractable on a small GPU — the full
            stride-256 grid is 361 windows per tile, which is 56k crops.
        min_positive: reject a window whose mask mean is below this, unless it
            survives the ``empty_ratio`` lottery. Inria is heavily forest and
            water (kitsap, tyrol-w especially), so unfiltered sampling wastes
            most of the compute on empty sky.
        empty_ratio: fraction of windows allowed through with no buildings, so
            the model still learns true negatives.
        augment: D4 + photometric, as in DataLoaderSegmentation.
        deterministic: tile a fixed grid instead of sampling randomly. Use for
            validation so the numbers are comparable epoch to epoch.
    """

    def __init__(
        self,
        image_files: list[Path],
        gt_dir: str | Path | None,
        window: int = INRIA_WINDOW,
        samples_per_tile: int = 16,
        min_positive: float = 0.005,
        empty_ratio: float = 0.1,
        augment: bool = True,
        stats_key: str = "imagenet",
        deterministic: bool = False,
        stride: int | None = None,
        seed: int = 0,
    ):
        if window % 32:
            raise ValueError(f"window must be a multiple of 32 (got {window})")

        self.image_files = [Path(p) for p in image_files]
        if not self.image_files:
            raise ValueError("image_files is empty")

        self.gt_dir = Path(gt_dir) if gt_dir is not None else None
        if self.gt_dir is not None:
            missing = [p.name for p in self.image_files
                       if not (self.gt_dir / p.name).exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} tiles have no ground truth in {self.gt_dir}, "
                    f"e.g. {missing[:3]}. Note Inria's own test cities ship without "
                    f"labels — score on a held-out slice of train/ instead."
                )

        self.window = window
        self.samples_per_tile = samples_per_tile
        self.min_positive = min_positive
        self.empty_ratio = empty_ratio
        self.augment = augment
        self.deterministic = deterministic
        self.stride = stride or window // 2
        self.seed = seed
        self.mean, self.std = norm_stats(stats_key)
        self.stats_key = stats_key

        self._cache: dict[Path, tuple] = {}

        if deterministic:
            self._grid = self._build_grid()

    # ------------------------------------------------------------------ #
    def _build_grid(self) -> list[tuple[int, int, int]]:
        """Fixed (tile_index, y, x) list covering each tile at ``stride``."""
        probe = open_tile(self.image_files[0])
        h, w = probe.shape[:2]
        ys = list(range(0, max(h - self.window, 0) + 1, self.stride)) or [0]
        xs = list(range(0, max(w - self.window, 0) + 1, self.stride)) or [0]
        return [(i, y, x) for i in range(len(self.image_files)) for y in ys for x in xs]

    def _tile(self, idx: int):
        path = self.image_files[idx]
        if path not in self._cache:
            img = open_tile(path)
            gt = open_tile(self.gt_dir / path.name) if self.gt_dir else None
            # A memmap costs no RAM; a PIL fallback costs 75 MB per tile, so cap
            # the cache to avoid quietly eating memory on the fallback path.
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[path] = (img, gt)
        return self._cache[path]

    def _sample_window(self, rng: random.Random, idx: int):
        img, gt = self._tile(idx)
        h, w = img.shape[:2]
        for _ in range(12):  # bounded retries; never loop forever on a blank tile
            y = rng.randrange(0, max(h - self.window, 1))
            x = rng.randrange(0, max(w - self.window, 1))
            if gt is None:
                return y, x
            m = gt[y : y + self.window, x : x + self.window]
            if (m > 127).mean() >= self.min_positive or rng.random() < self.empty_ratio:
                return y, x
        return y, x

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        if self.deterministic:
            return len(self._grid)
        return len(self.image_files) * self.samples_per_tile

    def __getitem__(self, index: int):
        if self.deterministic:
            idx, y, x = self._grid[index]
        else:
            # Seed per item so workers do not draw identical windows and the
            # epoch is reproducible.
            rng = random.Random((self.seed, index).__hash__())
            idx = index // self.samples_per_tile
            y, x = self._sample_window(rng, idx)

        img, gt = self._tile(idx)
        patch = np.ascontiguousarray(img[y : y + self.window, x : x + self.window, :3])
        mask = (
            np.ascontiguousarray(gt[y : y + self.window, x : x + self.window])
            if gt is not None
            else np.zeros(patch.shape[:2], dtype=np.uint8)
        )

        pim = Image.fromarray(patch)
        mim = Image.fromarray((mask > 127).astype(np.uint8) * 255, mode="L")

        if self.augment:
            ops = random.choice(D4_OPS)
            for op in ops:
                pim = pim.transpose(op)
                mim = mim.transpose(op)
            if random.random() < 0.5:
                pim = TF.adjust_brightness(pim, float(np.random.normal(1.0, 0.1)))
            if random.random() < 0.3:
                pim = TF.adjust_contrast(pim, float(np.random.uniform(0.85, 1.15)))

        t = TF.to_tensor(pim)
        t = TF.normalize(t, mean=self.mean, std=self.std)
        y_t = (torch.from_numpy(np.array(mim, dtype=np.uint8)) > 127).float()
        return t, y_t

    def __repr__(self) -> str:
        return (
            f"InriaWindowDataset(tiles={len(self.image_files)}, len={len(self)}, "
            f"window={self.window}, deterministic={self.deterministic}, "
            f"stats='{self.stats_key}')"
        )
