from __future__ import annotations

import pytest
import numpy as np
from pyproj import Transformer

from tile2net.raster.source import WashingtonDC
from tile2net.raster.tile import Tile


@pytest.fixture
def tile() -> Tile:
    """Return a deterministic z19 tile without network or filesystem I/O."""
    return Tile(
        xtile=154_372,
        ytile=197_054,
        idd=0,
        position=(0, 0),
        zoom=19,
    )


def test_tile2poly_uses_exact_tile_bounds(tile: Tile) -> None:
    polygon = tile.tile2poly()

    assert polygon.bounds == pytest.approx(
        (tile.left, tile.bottom, tile.right, tile.top)
    )


def test_tile2poly_uses_explicit_bounds(tile: Tile) -> None:
    bounds = (-77.1, 38.8, -77.0, 38.9)

    assert tile.tile2poly(*bounds).bounds == pytest.approx(bounds)


def test_transform_project_returns_xy_ordered_bounds(tile: Tile) -> None:
    expected = Transformer.from_crs(4326, 3857, always_xy=True).transform_bounds(
        tile.left,
        tile.bottom,
        tile.right,
        tile.top,
    )

    assert tile.transformProject(4326, 3857) == pytest.approx(expected)


def test_washington_dc_url_matches_declared_source(tile: Tile) -> None:
    from urllib.parse import parse_qs, urlsplit

    url = next(iter(WashingtonDC()[[tile]]))
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    expected_bbox = tile.transformProject(4326, 3857)
    actual_bbox = tuple(float(value) for value in query["bbox"][0].split(","))

    assert url.startswith(f"{WashingtonDC.server}/exportImage?")
    assert "Ortho_2023" in parsed.path
    assert "Ortho_2021" not in parsed.path
    assert actual_bbox == pytest.approx(expected_bbox)
    assert query["bboxSR"] == ["3857"]
    assert query["imageSR"] == ["3857"]
    assert query["size"] == ["512,512"]


def test_mask2poly_fills_small_holes_without_dataframe_apply(tile: Tile) -> None:
    """Exercise the production hole-filling branch with an RGB mask."""
    mask = np.zeros((64, 64, 3), dtype=np.uint8)
    mask[8:56, 8:56, 2] = 255
    mask[30:34, 30:34, 2] = 0

    result = tile.mask2poly(
        mask,
        class_name="sidewalk",
        class_id=2,
        class_hole_size=25,
        img_array=True,
    )

    assert result is not False
    assert not result.empty
    assert result.crs == tile.crs
    assert result.geometry.is_valid.all()
    assert all(not geometry.interiors for geometry in result.geometry.array)
