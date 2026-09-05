from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely

from tile2net.raster.grid import Grid
from tile2net.raster.raster import Raster

LOCATION = [40.7490, -74.0000, 40.7500, -73.9990]


@pytest.fixture
def grid() -> Grid:
    """Return a small deterministic grid without project or network I/O."""
    return Grid(
        name="test",
        location=LOCATION,
        zoom=19,
        padding=False,
    )


def test_grid_counts_every_active_tile(grid: Grid) -> None:
    assert grid.tiles.shape == (3, 3)
    assert grid.num_active == grid.tiles.size == grid.num_tiles


def test_make_inactive_deduplicates_tile_ids(grid: Grid) -> None:
    grid.make_inactive([0, 0, 4])

    assert grid.num_active == 7
    assert not grid.tiles.ravel()[0].active
    assert not grid.tiles.ravel()[4].active


@pytest.mark.parametrize("tile_id", [-1, 9])
def test_make_inactive_rejects_invalid_tile_ids(grid: Grid, tile_id: int) -> None:
    with pytest.raises(IndexError, match="outside the grid"):
        grid.make_inactive([tile_id])


def test_vectorized_grid_geometry_matches_tiles_exactly(grid: Grid) -> None:
    grid_gdf = grid.create_grid_gdf()
    expected_bounds = np.asarray(
        [[tile.left, tile.top, tile.right, tile.bottom] for tile in grid.tiles.ravel()],
        dtype=np.float64,
    )
    actual_bounds = grid_gdf[
        [
            "topleft_x",
            "topleft_y",
            "bottomright_x",
            "bottomright_y",
        ]
    ].to_numpy()
    expected_geometry = np.asarray(
        [tile.tile2poly() for tile in grid.tiles.ravel()],
        dtype=object,
    )

    np.testing.assert_array_equal(actual_bounds, expected_bounds)
    np.testing.assert_array_equal(
        shapely.to_wkb(grid_gdf.geometry.array),
        shapely.to_wkb(expected_geometry),
    )


def test_grid_array_cache_is_reused_but_not_exposed(grid: Grid) -> None:
    cached = grid._grid_array_cache
    first_geometry = cached["geometry"][0]

    exported = grid.create_grid_gdf()
    exported.at[0, "geometry"] = shapely.Point(0, 0)
    exported.at[0, "xtile"] = -1

    assert grid._grid_array_cache is cached
    assert shapely.equals(grid.create_grid_gdf().geometry.iloc[0], first_geometry)
    assert grid.create_grid_gdf().at[0, "xtile"] == grid.tiles.ravel()[0].xtile

    grid.update_tiles()
    assert grid._grid_array_cache is not cached


def test_unclipped_boundary_does_not_materialize_geodataframe(
    grid: Grid,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = gpd.GeoDataFrame(
        geometry=[grid.tiles.ravel()[4].tile2poly().buffer(-1e-7)],
        crs=grid.crs,
    )
    monkeypatch.setattr(
        grid,
        "get_boundary",
        lambda city=None, address=None, path=None: boundary,
    )

    def fail_if_called() -> None:
        pytest.fail("create_grid_gdf should not run for an unclipped mask")

    monkeypatch.setattr(grid, "create_grid_gdf", fail_if_called)

    grid.get_in_boundary()

    assert grid.active_tile_ids == [4]


def test_geoparquet_boundary_preserves_full_grid_and_filters_metadata(
    grid: Grid,
    tmp_path: Path,
) -> None:
    original_bbox = grid.bbox
    original_ids = [tile.idd for tile in grid.tiles.ravel()]
    center_tile = grid.tiles.ravel()[4]
    boundary_path = tmp_path / "boundary.parquet"
    gpd.GeoDataFrame(
        {"name": ["test boundary"]},
        geometry=[center_tile.tile2poly().buffer(-1e-7)],
        crs=grid.crs,
    ).to_parquet(boundary_path)

    bounded = Grid(
        name="bounded",
        location=LOCATION,
        zoom=19,
        padding=False,
        boundary_path=boundary_path,
    )

    assert bounded.tiles.shape == grid.tiles.shape == (3, 3)
    assert bounded.bbox == pytest.approx(original_bbox)
    assert [tile.idd for tile in bounded.tiles.ravel()] == original_ids
    assert [tile.idd for tile in bounded.tiles.ravel() if tile.active] == [4]
    assert bounded.num_active == bounded.num_inside == 1
    assert bounded._create_info_dict(df=True)["idd"].tolist() == [4]
    assert len(bounded.create_grid_gdf()) == bounded.num_tiles == 9

    clipped = bounded.get_in_boundary(path=boundary_path, clipped=True)
    assert clipped is not None
    assert clipped["idd"].tolist() == [4]
    assert shapely.equals(
        clipped.geometry.iloc[0],
        gpd.read_parquet(boundary_path).geometry.iloc[0],
    )


def test_raster_metadata_persists_mask_without_boundary_path(
    grid: Grid,
    tmp_path: Path,
) -> None:
    boundary_path = tmp_path / "private_boundary.parquet"
    gpd.GeoDataFrame(
        geometry=[grid.tiles.ravel()[4].tile2poly().buffer(-1e-7)],
        crs=grid.crs,
    ).to_parquet(boundary_path)
    raster = Raster(
        name="bounded",
        location=LOCATION,
        input_dir=tmp_path / "input" / "x_y.png",
        output_dir=tmp_path / "output",
        zoom=19,
        padding=False,
        boundary_path=boundary_path,
    )

    metadata = raster.save_info_json(return_dict=True, new_tstep=2)

    assert metadata["active_tile_ids"] == [0]
    assert "boundary_path" not in metadata
    json.dumps(metadata)
    restored = Raster.from_info(metadata)
    assert restored.tile_step == 2
    assert restored.tiles.shape == (2, 2)
    assert restored.active_tile_ids == [0]
    assert restored.num_active == restored.num_inside == 1


def test_save_ntw_polygons_accepts_empty_result(tmp_path: Path) -> None:
    grid = Grid(
        name="empty-polygons",
        location=LOCATION,
        output_dir=tmp_path,
        zoom=19,
        padding=False,
    )

    result = grid.save_ntw_polygons(gpd.GeoDataFrame())

    assert result.empty
    assert result.geometry.name == "geometry"
    assert result.crs == grid.crs
    assert result.columns.tolist() == ["f_type", "geometry"]
    assert grid.ntw_poly is result
    assert not tuple(grid.project.polygons.path.iterdir())


def test_save_ntw_polygons_defaults_to_geoparquet(tmp_path: Path) -> None:
    grid = Grid(
        name="vector-output",
        location=LOCATION,
        output_dir=tmp_path,
        zoom=19,
        padding=False,
    )
    polygons = gpd.GeoDataFrame(
        {"f_type": ["sidewalk", "crosswalk", "road"]},
        geometry=[
            shapely.box(0, 0, 20, 3),
            shapely.box(8, -3, 12, 6),
            shapely.box(0, -6, 20, -3),
        ],
        crs="EPSG:3857",
    )

    result = grid.save_ntw_polygons(polygons)
    path = grid.project.polygons.path / "vector-output-polygons.parquet"
    restored = gpd.read_parquet(path)

    assert not result.empty
    assert restored.crs == result.crs
    assert restored.geometry.equals(result.geometry)
    assert not tuple(grid.project.polygons.path.glob("*.shp"))
