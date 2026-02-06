from __future__ import annotations

import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import *
from pathlib import Path
from typing import *
from typing import Self, TypedDict

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from tile2net.core.cfg import cfg
from tile2net.core.cfg.logger import logger
from .dataloader import DataLoader
from .datawrapper import DataWrapper
from .stitch import StitchDataSet
from ..pednet import PedNet
from ..vecgrid.mask2poly import Mask2Poly

if TYPE_CHECKING:
    from tile2net.core.vecgrid import VecGrid


@contextmanager
def _silent_io() -> Iterator[None]:
    prev_disable: int = logging.root.manager.disable
    prev_tqdm: Optional[str] = os.environ.get('TQDM_DISABLE')
    try:
        logging.disable(logging.ERROR)
        os.environ['TQDM_DISABLE'] = '1'
        with open(os.devnull, 'w') as devnull, \
                redirect_stdout(devnull), \
                redirect_stderr(devnull):
            yield
    finally:
        logging.disable(prev_disable)
        if prev_tqdm is None:
            os.environ.pop('TQDM_DISABLE', None)
        else:
            os.environ['TQDM_DISABLE'] = prev_tqdm


def _worker_init(_: int) -> None:
    logging.disable(logging.ERROR)
    os.environ['TQDM_DISABLE'] = '1'
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)


ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class VecDataWrapper(
    DataWrapper
):
    @classmethod
    def from_segtiles(
            cls,
            *,
            static: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = None,
            force: bool | ArrayLike = True,
            affine: ArrayLike,
            lonmin: ArrayLike,
            latmin: ArrayLike,
            lonmax: ArrayLike,
            latmax: ArrayLike,
            polygon_file: ArrayLike,
            line_file: ArrayLike,
            **kwargs,
    ) -> Self:
        return super().from_columns(
            image_paths=static,
            index=index,
            row=row,
            col=col,
            background=background,
            force=force,
            affine=affine,
            lonmin=lonmin,
            latmin=latmin,
            lonmax=lonmax,
            latmax=latmax,
            polygon_file=polygon_file,
            line_file=line_file,
            **kwargs,
        )

    @cached_property
    def dataset(self) -> VecDataSet:
        return VecDataSet(self)


class VecResult(TypedDict):
    polygons: GeoDataFrame


class VecDataSet(
    StitchDataSet
):
    def __getitem__(self, item: int) -> int:

        empty_gdf = (
            gpd.GeoDataFrame(
                {"geometry": []},
                geometry="geometry",
                crs=3857,
            )
            .set_index(
                pd.Index([], name="feature"),
            )
        )
        with _silent_io():
            grayscale = super().__getitem__(item)
            affine = self.affine[item]
            xmin = self.lonmin[item]
            ymin = self.latmin[item]
            xmax = self.lonmax[item]
            ymax = self.latmax[item]
            polygons_file = self.polygon_file[item]
            network_file = self.network_file[item]
            polys = None
            Path(polygons_file).parent.mkdir(parents=True, exist_ok=True)
            Path(network_file).parent.mkdir(parents=True, exist_ok=True)
            grid_size = 0.1
            with cfg:
                try:
                    polys = Mask2Poly.from_array(grayscale, affine, crs=3857)
                    assert isinstance(polys, Mask2Poly)

                    if polys.empty:
                        # Persist the empty layers so downstream steps have concrete files
                        empty_gdf.to_parquet(polygons_file)
                        empty_gdf.to_parquet(network_file)

                        # msg = f'No polygons generated for {static}; wrote empty layers instead.'
                        msg = f'No polygons generated for {polygons_file}; wrote empty layers instead.'
                        logging.warning(msg)
                        return item

                    # net = PedNet.from_mask2poly(polys)
                    net = (
                        polys
                        .postprocess(
                            min_poly_area=cfg.polygon.min_polygon_area,
                            convexity=cfg.polygon.convexity,
                            simplify=cfg.polygon.simplify,
                            max_hole_area=cfg.polygon.max_hole_area,
                        )
                        .frame.pipe(PedNet.from_mask2poly)
                    )

                    if net.empty:
                        # Persist the empty layers so downstream steps have concrete files
                        empty_gdf.to_parquet(polygons_file)
                        empty_gdf.to_parquet(network_file)

                        # msg = f'No network generated for {static}; wrote empty layers instead.'
                        msg = f'No network generated for {polygons_file}; wrote empty layers instead.'
                        logging.warning(msg)
                        return item

                    # precompute lines and edges for clarity
                    lines = net.center.lines
                    edges = lines.edges

                    clipped = (
                        net.center.clipped
                        .frame
                        .pipe(
                            net.center.clipped.from_frame,
                            wrapper=net.center.clipped
                        )
                    )

                    polys = (
                        net.frame
                        .clip_by_rect(xmin, ymin, xmax, ymax)
                        .to_frame('geometry')
                        .set_precision(grid_size=grid_size)
                        .to_frame('geometry')
                    )

                    clipped = (
                        clipped.frame
                        .clip_by_rect(xmin, ymin, xmax, ymax)
                        .set_precision(grid_size=grid_size)
                        .to_frame('geometry')
                        .dissolve(by='feature')
                        .explode()
                    )

                    polys.to_parquet(polygons_file)
                    clipped.to_parquet(network_file)

                except Exception as e:
                    msg = (
                        'Error vectorizing:\n'
                        f'affine={affine!r},\n'
                    )
                    if isinstance(polys, gpd.GeoDataFrame):
                        msg += f'polygons_file={polygons_file}\n'
                        polys.to_parquet(polygons_file)
                    msg += f'{e}'
                    logger.error(msg)
                    raise

                finally:
                    for name in ("polys", "clipped", "net"):
                        if name in locals():
                            del locals()[name]
                    import gc
                    gc.collect()

        return item

    @cached_property
    def affine(self) -> list[object]:
        """
        Affine transformation parameters for georeferencing each tile.

        See also:
            >>> VecGrid.affine_params
        """
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .affine
            .first()
            .tolist()
        )
        return result

    @cached_property
    def lonmin(self) -> list[float]:
        """
        Minimum longitude of the tile.

        See also:
            >>> VecGrid.lonmin
        """
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .lonmin
            .first()
            .tolist()
        )
        return result

    @cached_property
    def latmin(self) -> list[float]:
        """
        Minimum latitude of the tile.

        See also:
            >>> VecGrid.latmin
        """
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .latmin
            .first()
            .tolist()
        )
        return result

    @cached_property
    def lonmax(self) -> list[float]:
        """
        Maximum longitude of the tile.

        See also:
            >>> VecGrid.lonmax
        """
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .lonmax
            .first()
            .tolist()
        )
        return result

    @cached_property
    def latmax(self) -> list[float]:
        """
        Maximum latitude of the tile.

        See also:
            >>> VecGrid.latmax
        """
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .latmax
            .first()
            .tolist()
        )
        return result

    @cached_property
    def polygon_file(self) -> list[str]:
        """
        File-paths to polygon geometry parquet files extracted from segmentation masks.

        See also:
            >>> VecGrid.file.polygons
        """
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .polygon_file
            .first()
            .tolist()
        )
        return result

    @cached_property
    def network_file(self) -> list[str]:
        """
        File-paths to line geometry parquet files extracted from segmentation masks.

        See also:
            >>> VecGrid.file.network
        """
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .line_file
            .first()
            .tolist()
        )
        return result


class VecDataLoader(
    DataLoader
):
    dataset: VecDataSet
    if False:
        def __iter__(self) -> Iterator[list[int]]:
            ...
