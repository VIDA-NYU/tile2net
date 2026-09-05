from __future__ import annotations

from tile2net.raster.manifest import filter_paths_by_tile_ids


def test_filter_paths_by_active_tile_ids() -> None:
    paths = ["0_0_0.png", "0_1_1.png", "1_0_2.png", "README.txt"]

    assert filter_paths_by_tile_ids(paths, [0, 2]) == [
        "0_0_0.png",
        "1_0_2.png",
    ]


def test_filter_paths_is_noop_without_active_ids() -> None:
    paths = ["0_0_0.png", "unrelated.file"]

    assert filter_paths_by_tile_ids(paths, None) == paths
