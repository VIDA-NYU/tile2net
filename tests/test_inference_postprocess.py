from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("cv2")
pytest.importorskip("tabulate")

from tile2net.raster.grid import Grid
from tile2net.raster.pednet import PedNet
from tile2net.tileseg.config import cfg
from tile2net.tileseg.inference.inference import (
    LocalDumper,
    build_pedestrian_network,
    read_segmentation_png,
)
from tile2net.tileseg.utils.misc import ThreadedDumper

LOCATION = [40.7490, -74.0000, 40.7500, -73.9990]


@pytest.mark.parametrize(
    ("active", "expected_calls"),
    [(True, 1), (False, 0)],
)
def test_black_pixels_do_not_override_tile_activity(
    active: bool,
    expected_calls: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use explicit tile activity instead of image darkness as a validity mask."""
    tile = SimpleNamespace(active=active)
    grid = SimpleNamespace(
        tiles=np.asarray([[tile]], dtype=object),
        pose_dict={0: (0, 0)},
    )
    dataset = SimpleNamespace(
        colorize_mask=lambda _: Image.new("RGB", (4, 4), color=(0, 0, 0))
    )
    was_immutable = cfg.is_immutable()
    cfg.immutable(False)
    monkeypatch.setitem(cfg, "DATASET_INST", dataset)
    monkeypatch.setitem(cfg, "RESULT_DIR", str(tmp_path))

    dumper = object.__new__(ThreadedDumper)
    dumper.args = SimpleNamespace(dump_percent=0)
    dumper.dump_percent = 100
    dumper.futures = []
    dumper.threads = ThreadPoolExecutor(max_workers=1)
    dumper.inv_normalize = lambda image: image
    dumper.visualize = lambda image: image
    dumper.save_prob_and_err_mask = lambda *args: (False, None)
    dumper.create_composite_image = lambda *args: None
    dumper.get_dump_assets = lambda *args: None
    calls: list[np.ndarray] = []
    dumper.map_features = (
        lambda _tile, image, img_array=True, save_segmentation=False:
        calls.append(image)
    )
    dump = {
        "input_images": torch.zeros((1, 3, 4, 4)),
        "gt_images": torch.zeros((1, 4, 4), dtype=torch.int64),
        "img_names": ["stitched_0"],
        "assets": {"predictions": np.zeros((1, 4, 4), dtype=np.uint8)},
    }

    try:
        assert list(dumper.dump(dump, 0, testing=True, grid=grid)) == []
    finally:
        dumper.threads.shutdown(wait=True)
        cfg.immutable(was_immutable)

    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    ("dump_percent", "expected_saved"),
    [(0, False), (100, True)],
)
def test_dump_percent_controls_viewable_prediction_output(
    tmp_path: Path,
    dump_percent: int,
    expected_saved: bool,
) -> None:
    prediction = np.zeros((4, 4, 3), dtype=np.uint8)
    prediction[1:3, 1:3] = (255, 0, 0)
    segmentation_path = tmp_path / "prediction.png"
    tile = SimpleNamespace(
        segmentation=segmentation_path,
        mask2poly=lambda *args, **kwargs: False,
    )
    dumper = object.__new__(LocalDumper)
    dumper.args = SimpleNamespace(dump_percent=dump_percent)
    dumper.dump_percent = 100
    dumper.save_dir = str(tmp_path / "seg_results")
    dumper.futures = []
    dumper.threads = ThreadPoolExecutor(max_workers=1)

    try:
        selected = dumper.create_composite_image(
            Image.new("RGB", (4, 4)),
            Image.fromarray(prediction),
            "prediction",
        )
        result = dumper.map_features(
            tile,
            prediction,
            img_array=True,
            save_segmentation=selected,
        )
    finally:
        dumper.threads.shutdown(wait=True)

    assert result is None
    assert segmentation_path.exists() is expected_saved
    if expected_saved:
        np.testing.assert_array_equal(
            read_segmentation_png(segmentation_path),
            prediction,
        )
    assert not tuple(tmp_path.glob("*.npy"))


def test_empty_polygon_result_skips_network_creation(tmp_path: Path) -> None:
    grid = Grid(
        name="empty-network",
        location=LOCATION,
        output_dir=tmp_path,
        zoom=19,
        padding=False,
    )

    network = build_pedestrian_network(grid, gpd.GeoDataFrame())

    assert network is None
    assert grid.ntw_poly.empty
    assert not grid.project.network.path.exists()


def test_network_output_defaults_to_geoparquet(tmp_path: Path) -> None:
    grid = Grid(
        name="network-output",
        location=LOCATION,
        output_dir=tmp_path,
        zoom=19,
        padding=False,
    )
    network = gpd.GeoDataFrame(
        {"f_type": ["sidewalk"]},
        geometry=gpd.points_from_xy([-74.0], [40.75]),
        crs="EPSG:4326",
    )
    builder = PedNet(poly=gpd.GeoDataFrame(), project=grid.project)

    path = builder.save_network(network)
    restored = gpd.read_parquet(path)

    assert path.name == "network-output-network.parquet"
    assert restored.crs == network.crs
    assert restored.geometry.equals(network.geometry)
    assert not tuple(grid.project.network.path.glob("*.shp"))
