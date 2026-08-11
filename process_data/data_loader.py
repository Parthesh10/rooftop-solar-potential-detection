"""Paired image/mask dataset for binary rooftop segmentation.

Rewritten to fix, in order of severity (see potential-fixes.md):

* **F-04** — images and masks were globbed independently and paired by list
  position. ``glob.glob`` is unsorted and OS-dependent, so one missing file
  silently misaligned every subsequent pair with no error. Masks are now derived
  from the image filename and their existence is verified at construction.
* **F-07** — training cropped to 248x248 while val/test stayed at 250x250, and
  neither divides by 16 (the U-Net's downsample factor). Training now crops to
  a configurable multiple of 32; evaluation keeps the native size and the model
  boundary pads (see utils.pad_to_multiple).
* **F-13** — the mask was opened without ``.convert()``, so a palette-mode PNG
  would yield palette *indices* from channel 0.
* **F-14** — masks went through bilinear-capable rotation. Augmentation is now
  the exact 8-element dihedral (D4) group via lossless PIL transposes, sampled
  uniformly rather than as three independent coin flips.
* **F-20** — ``change_hsv`` (a per-pixel Python double loop), ``flip``,
  ``add_noise``, ``add_uniform_noise``, ``add_gaussian_noise``,
  ``ceil_floor_image`` and ``normalization2`` were dead code, and the two noise
  helpers returned unclamped float64 because they clamped the wrong array.
  All removed.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import transforms

from config import TRAIN_CROP, norm_stats

__all__ = ["DataLoaderSegmentation", "D4_OPS"]


# The eight elements of the dihedral group D4, expressed as lossless PIL
# transposes. Every one is an exact pixel permutation: no interpolation, so a
# mask survives unchanged (F-14).
D4_OPS: tuple[tuple[int, ...], ...] = (
    (),                                                        # identity
    (Image.ROTATE_90,),
    (Image.ROTATE_180,),
    (Image.ROTATE_270,),
    (Image.FLIP_LEFT_RIGHT,),
    (Image.FLIP_TOP_BOTTOM,),
    (Image.TRANSPOSE,),                                        # main diagonal
    (Image.TRANSVERSE,),                                       # anti-diagonal
)


def _apply_d4(img: Image.Image, ops: tuple[int, ...]) -> Image.Image:
    for op in ops:
        img = img.transpose(op)
    return img


class DataLoaderSegmentation(data.Dataset):
    """Aerial image + binary mask pairs.

    Args:
        folder_path_img: directory of RGB ``.png`` tiles.
        folder_path_mask: directory of binary mask ``.png`` tiles. ``None`` for
            unlabelled inference sets, in which case an all-zero mask is
            returned so the batch shape stays uniform.
        augment: apply D4 + photometric augmentation and random cropping.
            Set False for val/test.
        crop: side length of the random crop applied when ``augment`` is True.
            Must be a multiple of 32 (see F-07). ``None`` disables cropping.
        stats_key: which normalisation constants to use, from ``config.NORM_STATS``.
        mask_suffix: filename suffix that turns an image stem into its mask stem.
            The Swiss DOP25 set uses ``"_label"``; pass ``""`` for datasets whose
            masks share the image filename.
        strict: raise if any mask is missing. Leave True — a dataset that cannot
            verify its own pairing should not be allowed to start training.
    """

    def __init__(
        self,
        folder_path_img: str | Path | None = None,
        folder_path_mask: str | Path | None = None,
        augment: bool = True,
        crop: int | None = TRAIN_CROP,
        stats_key: str = "all",
        mask_suffix: str = "_label",
        strict: bool = True,
        img_files: list[Path] | None = None,
    ):
        # Two construction modes:
        #   * a directory pair (the original, for data/<split>/{images,labels})
        #   * an explicit file list (for a geographic split that pools tiles from
        #     several directories — see from_files / F-05). The file-list mode
        #     resolves each mask next to its own image, so a re-split costs no
        #     disk at all instead of duplicating 170 MB.
        if img_files is not None:
            self.image_dir = None
            self.img_files = [Path(p) for p in img_files]
            if not self.img_files:
                raise ValueError("img_files is empty")
            self.has_masks = folder_path_mask is not None or strict
            mask_dirs = [
                Path(folder_path_mask) if folder_path_mask is not None
                else p.parent.parent / "labels"
                for p in self.img_files
            ]
        else:
            if folder_path_img is None:
                raise ValueError("pass either folder_path_img or img_files")
            self.image_dir = Path(folder_path_img)
            if not self.image_dir.is_dir():
                raise FileNotFoundError(f"image directory not found: {self.image_dir}")

            # sorted(), not glob order — deterministic across filesystems (F-04).
            self.img_files = sorted(self.image_dir.glob("*.png"))
            if not self.img_files:
                raise FileNotFoundError(f"no .png files in {self.image_dir}")

            self.has_masks = folder_path_mask is not None
            if self.has_masks:
                mask_dir = Path(folder_path_mask)
                if not mask_dir.is_dir():
                    raise FileNotFoundError(f"mask directory not found: {mask_dir}")
                mask_dirs = [mask_dir] * len(self.img_files)

        self.mask_files: list[Path] | None = None

        if self.has_masks:
            self.mask_files = [
                self._mask_for(p, d, mask_suffix) for p, d in zip(self.img_files, mask_dirs)
            ]

            missing = [
                img for img, m in zip(self.img_files, self.mask_files) if m is None
            ]
            if missing and strict:
                raise FileNotFoundError(
                    f"{len(missing)} of {len(self.img_files)} images have no matching mask "
                    f"(tried '<stem>{mask_suffix}.png' and '<stem>.png'). "
                    f"First few: {[p.name for p in missing[:3]]}"
                )
            if missing:  # non-strict: drop the unpaired images rather than misalign
                keep = [i for i, m in enumerate(self.mask_files) if m is not None]
                self.img_files = [self.img_files[i] for i in keep]
                self.mask_files = [self.mask_files[i] for i in keep]

        self.augment = augment
        self.crop = crop
        if crop is not None and crop % 32 != 0:
            raise ValueError(f"crop must be a multiple of 32 (got {crop}) — see F-07")

        self.stats_key = stats_key
        self.mean, self.std = norm_stats(stats_key)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_files(cls, img_files, **kwargs) -> "DataLoaderSegmentation":
        """Build a dataset from an explicit list of image paths.

        Each mask is resolved next to its own image (``<parent>/../labels/``),
        so tiles pooled from several split directories work without moving or
        copying anything. This is how a geographic re-split (F-05) is consumed.
        """
        return cls(img_files=list(img_files), **kwargs)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, root: str | Path, **kwargs):
        """Build a dataset from a ``split.py`` manifest of root-relative paths."""
        root = Path(root)
        lines = Path(manifest_path).read_text(encoding="utf-8").split("\n")
        return cls.from_files([root / ln.strip() for ln in lines if ln.strip()], **kwargs)

    @staticmethod
    def _mask_for(img_path: Path, mask_dir: Path, suffix: str) -> Path | None:
        """Derive the mask path from the image path. Never positional (F-04)."""
        for candidate in (
            mask_dir / f"{img_path.stem}{suffix}.png",
            mask_dir / f"{img_path.stem}.png",
        ):
            if candidate.exists():
                return candidate
        return None

    # ------------------------------------------------------------------ #
    def augmentor(self, image: Image.Image, mask: Image.Image):
        """Geometric + photometric augmentation.

        Geometry is applied identically to image and mask; photometric changes
        touch the image only.
        """
        # Uniform over all 8 dihedral transforms. The old code used three
        # independent coin flips, which reaches the same 8 states but with
        # non-uniform probability (F-14).
        ops = random.choice(D4_OPS)
        image = _apply_d4(image, ops)
        mask = _apply_d4(mask, ops)

        if self.crop is not None:
            w, h = image.size
            if min(w, h) < self.crop:
                raise ValueError(
                    f"crop={self.crop} exceeds image size {w}x{h} for this dataset"
                )
            i, j, ch, cw = transforms.RandomCrop.get_params(
                image, output_size=(self.crop, self.crop)
            )
            image = TF.crop(image, i, j, ch, cw)
            mask = TF.crop(mask, i, j, ch, cw)

        # Photometric — real probability gates. The originals read
        # `if random.random() > 0:`, which is always True (F-07).
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, float(np.random.normal(1.0, 0.1)))
        if random.random() < 0.3:
            image = TF.adjust_contrast(image, float(np.random.uniform(0.85, 1.15)))
        if random.random() < 0.2:
            image = TF.gaussian_blur(image, 3, sigma=float(np.random.uniform(0.01, 0.6)))

        return image, mask

    def transform(self, image: Image.Image, mask: Image.Image):
        """PIL -> normalised CHW float tensor, and PIL -> {0,1} HW float tensor.

        This is the *only* place normalisation happens for training data;
        ``infer.preprocess`` mirrors it exactly for inference, and
        ``tests/test_preprocess_parity.py`` asserts the two agree (F-01).
        """
        x = TF.to_tensor(image)  # (3, H, W) in [0, 1]
        x = TF.normalize(x, mean=self.mean, std=self.std)

        y = torch.from_numpy(np.array(mask, dtype=np.uint8))  # (H, W), already 'L'
        y = (y > 127).float()
        return x, y

    # ------------------------------------------------------------------ #
    def __getitem__(self, index: int, show_og: bool = False):
        image = Image.open(self.img_files[index]).convert("RGB")

        if self.has_masks:
            # F-13: force greyscale so palette / RGBA PNGs cannot leak palette
            # indices or an alpha channel into channel 0.
            mask = Image.open(self.mask_files[index]).convert("L")
            if mask.size != image.size:
                raise ValueError(
                    f"size mismatch for {self.img_files[index].name}: "
                    f"image {image.size} vs mask {mask.size}"
                )
        else:
            mask = Image.new("L", image.size, 0)

        if self.augment:
            image, mask = self.augmentor(image, mask)

        original = image
        x, y = self.transform(image, mask)
        return (x, y, original) if show_og else (x, y)

    def __len__(self) -> int:
        return len(self.img_files)

    def __repr__(self) -> str:
        src = f"dir='{self.image_dir}'" if self.image_dir else "source=file-list"
        return (
            f"DataLoaderSegmentation(n={len(self)}, {src}, "
            f"masks={self.has_masks}, augment={self.augment}, crop={self.crop}, "
            f"stats='{self.stats_key}')"
        )
