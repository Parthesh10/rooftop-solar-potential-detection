"""The Indian fine-tuning pipeline: tile selection, label conversion, freezing.

Nothing here touches the network or trains a real model — that would make the
suite slow and flaky for no benefit. What is tested is the part a bug would
actually hide in: window scoring math, the labelme -> mask conversion, and the
encoder freeze/unfreeze that scripts/finetune_indian.py's two-phase training
depends on.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from process_data.labelme_to_masks import convert_dir, convert_one, polygons_to_mask
from scripts.finetune_indian import set_encoder_trainable, split_files
from scripts.select_finetune_tiles import Candidate, score_windows, select_top

pytest.importorskip("segmentation_models_pytorch")


# --------------------------------------------------------------------------- #
# labelme -> mask conversion
# --------------------------------------------------------------------------- #
def _square_shape(x0, y0, size, label="building"):
    return {"label": label, "shape_type": "polygon",
            "points": [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]]}


def test_polygons_to_mask_rasterises_a_building():
    mask = polygons_to_mask([_square_shape(10, 10, 20)], height=64, width=64)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert mask[20, 20] == 255          # inside the square
    assert mask[5, 5] == 0              # outside it


def test_polygons_to_mask_skips_excluded_labels():
    mask = polygons_to_mask([_square_shape(0, 0, 30, label="not-building")],
                            height=64, width=64)
    assert not mask.any()


def test_polygons_to_mask_skips_degenerate_shapes():
    tiny = {"label": "building", "shape_type": "polygon", "points": [[1, 1], [2, 2]]}
    mask = polygons_to_mask([tiny], height=32, width=32)
    assert not mask.any()


def test_polygons_to_mask_union_of_multiple_buildings():
    shapes = [_square_shape(0, 0, 10), _square_shape(20, 20, 10)]
    mask = polygons_to_mask(shapes, height=64, width=64)
    assert mask[5, 5] == 255
    assert mask[25, 25] == 255
    assert mask[15, 15] == 0            # the gap between them


def _write_labelme(tmp_path, stem: str, size: int, shapes: list[dict]):
    img = Image.fromarray(np.full((size, size, 3), 200, dtype=np.uint8))
    img.save(tmp_path / f"{stem}.png")
    (tmp_path / f"{stem}.json").write_text(json.dumps({
        "version": "5.4.1", "shapes": shapes,
        "imageHeight": size, "imageWidth": size,
        "imagePath": f"{stem}.png",
    }), encoding="utf-8")


def test_convert_one_writes_paired_image_and_mask(tmp_path):
    _write_labelme(tmp_path, "a", 64, [_square_shape(10, 10, 20)])
    out_img, out_lbl = tmp_path / "out" / "images", tmp_path / "out" / "labels"
    ok = convert_one(tmp_path / "a.json", tmp_path, out_img, out_lbl)
    assert ok is True
    assert (out_img / "a.png").exists()
    assert (out_lbl / "a_label.png").exists()
    mask = np.array(Image.open(out_lbl / "a_label.png"))
    assert mask.max() == 255


def test_convert_one_skips_an_unlabelled_candidate(tmp_path):
    _write_labelme(tmp_path, "empty", 64, [])
    ok = convert_one(tmp_path / "empty.json", tmp_path,
                     tmp_path / "out/images", tmp_path / "out/labels")
    assert ok is False
    assert not (tmp_path / "out").exists()


def test_convert_one_raises_a_clear_error_without_the_source_image(tmp_path):
    (tmp_path / "orphan.json").write_text(json.dumps({
        "shapes": [_square_shape(0, 0, 5)], "imageHeight": 32, "imageWidth": 32,
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no matching image"):
        convert_one(tmp_path / "orphan.json", tmp_path,
                   tmp_path / "out/images", tmp_path / "out/labels")


def test_convert_dir_reports_converted_and_skipped(tmp_path):
    _write_labelme(tmp_path, "has_building", 64, [_square_shape(5, 5, 10)])
    _write_labelme(tmp_path, "blank", 64, [])
    summary = convert_dir(tmp_path, tmp_path, tmp_path / "out")
    assert summary["converted"] == ["has_building.json"]
    assert summary["skipped"] == ["blank.json"]
    assert (summary["out_images"] / "has_building.png").exists()


def test_convert_dir_refuses_an_empty_annotation_directory(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="no .json annotations"):
        convert_dir(tmp_path / "images", tmp_path, tmp_path / "out")


# --------------------------------------------------------------------------- #
# tile selection: window scoring
# --------------------------------------------------------------------------- #
class _FakeGrid:
    """pixel_to_lonlat as a trivial affine map — enough to test window math."""

    def pixel_to_lonlat(self, x, y):
        return (77.0 + x * 1e-6, 13.0 - y * 1e-6)


def test_score_windows_flags_uniform_regions_as_not_built_up():
    h = w = 512
    mosaic = np.full((h, w, 3), 120, dtype=np.uint8)   # a flat field
    probs = np.full((h, w), 0.02, dtype=np.float32)
    cands = score_windows(mosaic, probs, _FakeGrid(), "aoi", window=512)
    assert len(cands) == 1
    assert cands[0].built_up < 1.0


def test_score_windows_uncertainty_is_maximal_at_p_half():
    h = w = 512
    mosaic = (np.random.default_rng(0).integers(0, 255, (h, w, 3))).astype(np.uint8)
    confident = np.full((h, w), 0.98, dtype=np.float32)
    uncertain = np.full((h, w), 0.5, dtype=np.float32)
    c_confident = score_windows(mosaic, confident, _FakeGrid(), "a", window=512)[0]
    c_uncertain = score_windows(mosaic, uncertain, _FakeGrid(), "a", window=512)[0]
    assert c_uncertain.uncertainty > c_confident.uncertainty
    assert c_confident.uncertainty == pytest.approx(0.02, abs=1e-6)
    assert c_uncertain.uncertainty == pytest.approx(0.5, abs=1e-6)


def test_score_windows_bounds_are_sane_lon_lat_order():
    mosaic = np.zeros((512, 512, 3), dtype=np.uint8)
    probs = np.zeros((512, 512), dtype=np.float32)
    c = score_windows(mosaic, probs, _FakeGrid(), "a", window=512)[0]
    west, south, east, north = c.bounds
    assert west < east
    assert south < north


def _make_candidates(n: int, seed: int = 0) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(Candidate(
            aoi="a", idx=i, x0=0, y0=0,
            built_up=float(rng.uniform(0, 30)),
            uncertainty=float(rng.uniform(0, 0.5)),
            mean_prob=float(rng.uniform(0, 1)),
            bounds=(0.0, 0.0, 0.001, 0.001),
        ))
    return out


def test_select_top_drops_windows_below_the_built_up_floor():
    cands = _make_candidates(50)
    for c in cands[:10]:
        c.built_up = 0.0          # force these below any reasonable floor
    chosen = select_top(cands, n=50, min_built_up=5.0, uncertain_fraction=0.5)
    assert all(c.built_up >= 5.0 for c in chosen)


def test_select_top_biases_toward_uncertainty_but_keeps_some_confident_ones():
    cands = _make_candidates(100)
    for c in cands:
        c.built_up = 50.0          # everything passes the built-up filter
    chosen = select_top(cands, n=40, min_built_up=1.0, uncertain_fraction=0.75)
    assert len(chosen) == 40
    mean_chosen = np.mean([c.uncertainty for c in chosen])
    mean_pool = np.mean([c.uncertainty for c in cands])
    assert mean_chosen > mean_pool
    # Not every pick is from the uncertain tail — some low-uncertainty windows
    # are kept too, or a fine-tune only ever sees the model's blind spots.
    assert min(c.uncertainty for c in chosen) < np.percentile(
        [c.uncertainty for c in cands], 25)


def test_select_top_returns_nothing_when_the_whole_pool_fails_the_filter():
    cands = _make_candidates(20)
    for c in cands:
        c.built_up = 0.0
    assert select_top(cands, n=10, min_built_up=100.0, uncertain_fraction=0.5) == []


# --------------------------------------------------------------------------- #
# fine-tuning: split + encoder freeze
# --------------------------------------------------------------------------- #
def test_split_files_refuses_too_few_tiles(tmp_path):
    (tmp_path / "images").mkdir()
    for i in range(5):
        (tmp_path / "images" / f"t{i}.png").write_bytes(b"")
    with pytest.raises(SystemExit, match="need at least"):
        split_files(tmp_path / "images", val_fraction=0.15, seed=0)


def test_split_files_is_deterministic_and_disjoint(tmp_path):
    (tmp_path / "images").mkdir()
    for i in range(30):
        (tmp_path / "images" / f"t{i:02d}.png").write_bytes(b"")
    train_a, val_a = split_files(tmp_path / "images", val_fraction=0.2, seed=0)
    train_b, val_b = split_files(tmp_path / "images", val_fraction=0.2, seed=0)
    assert train_a == train_b and val_a == val_b            # deterministic
    assert set(train_a).isdisjoint(set(val_a))               # no leakage
    assert len(train_a) + len(val_a) == 30
    assert len(val_a) == 6


def test_set_encoder_trainable_freezes_and_unfreezes_only_the_encoder():
    from model.registry import build_model

    model = build_model("unet++", "efficientnet-b0", encoder_weights=None)
    n_frozen = set_encoder_trainable(model, trainable=False)
    assert n_frozen > 0
    assert all(not p.requires_grad for p in model.encoder.parameters())
    # The decoder/head must be untouched by freezing the encoder.
    non_encoder = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    assert any(p.requires_grad for p in non_encoder)

    set_encoder_trainable(model, trainable=True)
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_set_encoder_trainable_raises_on_a_model_with_no_encoder():
    from model.unet import UNet

    with pytest.raises(AttributeError, match="no .encoder"):
        set_encoder_trainable(UNet(3, 1, False), trainable=False)
