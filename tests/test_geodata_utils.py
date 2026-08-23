from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from tile2net.raster.formats import VectorFormat
from tile2net.raster.tile_utils import geodata_utils


def test_unary_multi_repairs_geometry_with_copy_on_write() -> None:
    """Repair invalid geometry without relying on chained assignment."""
    invalid = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    valid = Polygon([(3, 0), (4, 0), (4, 1), (3, 1), (3, 0)])
    frame = gpd.GeoDataFrame(
        {"feature": ["invalid", "valid"], "shape": [invalid, valid]},
        geometry="shape",
        crs="EPSG:3857",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ChainedAssignmentError)
        result = geodata_utils.unary_multi(frame)

    assert result.geometry.name == "shape"
    assert result.crs == frame.crs
    assert result.geometry.is_valid.all()


def test_unary_multi_skips_repair_for_valid_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid an unnecessary make-valid operation when every geometry is valid."""
    valid = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    frame = gpd.GeoDataFrame(geometry=[valid], crs="EPSG:3857")

    def unexpected_make_valid(*args: object, **kwargs: object) -> None:
        raise AssertionError("make_valid must not run for valid geometry")

    monkeypatch.setattr(geodata_utils.shapely, "make_valid", unexpected_make_valid)

    result = geodata_utils.unary_multi(frame)

    assert result.geometry.is_valid.all()


def test_write_geoparquet_round_trip_is_atomic(tmp_path: Path) -> None:
    """Persist one portable geospatial file without leaving temporary output."""
    frame = gpd.GeoDataFrame(
        {"f_type": ["sidewalk"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])],
        crs="EPSG:4326",
    )
    destination = tmp_path / "polygons.parquet"

    result = geodata_utils.write_geoparquet(frame, destination)
    restored = gpd.read_parquet(result)

    assert result == destination
    assert restored.crs == frame.crs
    assert restored.geometry.equals(frame.geometry)
    assert restored["f_type"].tolist() == ["sidewalk"]
    assert tuple(tmp_path.iterdir()) == (destination,)


def test_write_vector_supports_optional_shapefile(tmp_path: Path) -> None:
    """Write Shapefile only when explicitly selected for compatibility."""
    frame = gpd.GeoDataFrame(
        {"f_type": ["sidewalk"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])],
        crs="EPSG:4326",
    )

    path = geodata_utils.write_vector(
        frame,
        tmp_path / "polygons",
        VectorFormat.SHAPEFILE,
    )
    restored = gpd.read_file(path)

    assert path.suffix == ".shp"
    assert restored.crs == frame.crs
    assert restored.geometry.iloc[0].equals(frame.geometry.iloc[0])
