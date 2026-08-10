"""Tests for leakage-free splitting (F-05)."""

from pathlib import Path

import pytest

from process_data.split import (
    block_split,
    city_holdout_split,
    inria_official_split,
    parse_lv03,
)


def _swiss(e: float, n: float) -> Path:
    return Path(f"DOP25_LV03_1301_11_2015_1_15_{e}_{n}.png")


def test_parse_lv03():
    assert parse_lv03(_swiss(497812.5, 120937.5)) == (497812.5, 120937.5)
    assert parse_lv03("DOP25_LV03_1301_11_2015_1_15_497812.5_120937.5_label.png") == (
        497812.5,
        120937.5,
    )
    assert parse_lv03("austin1.tif") is None


def test_block_split_never_separates_neighbours():
    """The core guarantee: no tile ends up in a different split from its block."""
    files = [
        _swiss(497000 + i * 62.5, 119000 + j * 62.5)
        for i in range(24)
        for j in range(24)
    ]
    split = block_split(files, block_size=1000.0, seed=0)

    assigned = {}
    for name, group in split.items():
        for f in group:
            assigned[f.name] = name

    for f in files:
        e, n = parse_lv03(f)
        block = (int(e // 1000), int(n // 1000))
        for g in files:
            ge, gn = parse_lv03(g)
            if (int(ge // 1000), int(gn // 1000)) == block:
                assert assigned[f.name] == assigned[g.name], (
                    f"{f.name} and {g.name} share a block but landed in "
                    f"{assigned[f.name]} and {assigned[g.name]}"
                )


def test_block_split_covers_every_file_exactly_once():
    files = [_swiss(497000 + i * 62.5, 119000 + j * 62.5) for i in range(12) for j in range(12)]
    split = block_split(files, seed=1)
    total = [f for group in split.values() for f in group]
    assert len(total) == len(files)
    assert {f.name for f in total} == {f.name for f in files}


def test_block_split_refuses_unparseable_names():
    """A silent fallback to random splitting would reintroduce the leak."""
    files = [_swiss(497000, 119000), Path("random_tile.png")]
    with pytest.raises(ValueError, match="no LV03 coordinates"):
        block_split(files)


def test_inria_official_protocol():
    files = [Path(f"{city}{i}.tif") for city in ("austin", "vienna") for i in range(1, 37)]
    split = inria_official_split(files)
    assert len(split["val"]) == 10   # 5 per city
    assert len(split["train"]) == 62
    assert {f.name for f in split["val"]} == {
        f"{c}{i}.tif" for c in ("austin", "vienna") for i in range(1, 6)
    }


def test_inria_split_handles_hyphenated_city_names():
    files = [Path(f"tyrol-w{i}.tif") for i in range(1, 37)]
    split = inria_official_split(files)
    assert len(split["val"]) == 5
    assert len(split["train"]) == 31


def test_city_holdout_isolates_one_city():
    files = [Path(f"{c}{i}.tif") for c in ("austin", "chicago", "vienna") for i in range(1, 6)]
    split = city_holdout_split(files, "chicago")
    assert all("chicago" in f.name for f in split["val"])
    assert not any("chicago" in f.name for f in split["train"])


def test_city_holdout_rejects_an_unknown_city():
    files = [Path("austin1.tif")]
    with pytest.raises(ValueError, match="no tiles found"):
        city_holdout_split(files, "bhopal")
