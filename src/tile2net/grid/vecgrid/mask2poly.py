from __future__ import annotations
import PIL.Image

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import *

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import rasterio.features
import rasterio.features as _rf
import shapely
import shapely.wkt
import skimage
import skimage.io  # assume already in deps
from PIL import Image
from affine import Affine
from numpy import ndarray
from pandas import Series
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import shapes
from shapely.geometry import shape
import io
from math import ceil
import matplotlib.pyplot as plt

from tile2net.grid.cfg.logger import logger
from ..benchmark import benchmark
from ..cfg import cfg
from ..frame.framewrapper import FrameWrapper

os.environ['USE_PYGEOS'] = '0'


class Mask2Poly(
    FrameWrapper,
):

    @classmethod
    def from_path(
            cls,
            path: str | Path,
            affine: Affine,
            crs: int = 3857,
    ) -> Self:
        original_max = Image.MAX_IMAGE_PIXELS

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', Image.DecompressionBombWarning)
            Image.MAX_IMAGE_PIXELS = None  # avoid the error variant
            array = skimage.io.imread(str(path))

        Image.MAX_IMAGE_PIXELS = original_max  # restore global state
        result = cls.from_array(array, affine, crs=crs)
        return result

    @classmethod
    def from_parquets(cls, files: Iterable[str | Path], ) -> Self:
        raise NotImplementedError
        paths = [str(Path(p)) for p in files]
        if not paths:
            return cls()

        logger.debug(f"Reading {len(paths)} parquet files into {cls.__name__} via pyarrow.dataset")
        # table = ds.dataset(paths, format="parquet").to_table(use_threads=True)
        # logger.debug(f"Arrow table rows={table.num_rows}, cols={table.num_columns}")

        # df = table.to_pandas(split_blocks=True, self_destruct=True)
        # geometry = gpd.GeoSeries.from_wkb(df.pop("geometry"), crs=4326)
        # gdf = cls(df, geometry=geometry, crs=4326)

        gdf = (
            # table
            ds.dataset(paths, format="parquet")
            .to_table(use_threads=True)
            .to_pandas(split_blocks=True, self_destruct=True)
            .geometry.pipe(gpd.GeoSeries.from_wkb, crs=4326)
            .to_frame(name='geometry')
            .pipe(cls)
        )

        logger.debug(f"GeoDataFrame assembled with {len(gdf)} row(s)")
        return gdf

    @classmethod
    def from_dir(
            cls,
            directory: str | Path,
            *,
            recursive: bool = True,
            threads: int | None = None,
            **read_parquet_kwargs,
    ) -> Self:
        base = Path(directory)
        pattern = '**/*.parquet' if recursive else '*.parquet'
        files = list(base.glob(pattern))
        if not files:
            msg = f"No parquet files found in {directory} with pattern {pattern}"
            raise FileNotFoundError(msg)
        return cls.from_parquets(files, threads=threads, **read_parquet_kwargs)

    @staticmethod
    def shapes_strict(
            arr: np.ndarray,
            mask: np.ndarray | None = None,
            transform: Affine | None = None,
            connectivity: int = 4,
    ) -> Iterable[tuple[dict, int]]:
        # Fail fast if the raster lacks geotransform metadata

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='error',
                category=NotGeoreferencedWarning,
            )
            try:
                yield from shapes(
                    arr,
                    mask=mask,
                    connectivity=connectivity,
                    transform=transform,
                )
            except NotGeoreferencedWarning as exc:
                raise RuntimeError(
                    'Input raster is not georeferenced; aborting.'
                ) from exc

    @classmethod
    @contextmanager
    def shapes_strict_context(self):
        original = _rf.shapes
        _rf.shapes = self.shapes_strict  # <-- your wrapper defined earlier
        try:
            yield
        finally:
            _rf.shapes = original  # always restore

    @classmethod
    def from_array(
            cls,
            array: np.ndarray,
            affine: Affine,
            crs: int = 3857,
    ) -> Self:
        # todo: check over where the crs whould be what, and to support user input
        """
        array:
            raw prediction array
        affine:
            affine transformation for tile
        crs:
            CRS of the output GeoDataFrame
        label2id:
            feature label to channel id mapping e.g. {'road': 0, 'sidewalk': 1, ...}
        """
        ARRAY = array
        concat: list[gpd.GeoDataFrame] = []
        label2id = cfg.label2id
        if array.ndim == 3:
            array = array[..., 0]  # assuming single channel for simplicity

        for label, id in label2id.items():
            # mask = np.array(array == id, dtype=np.uint8)
            # it = rasterio.features.shapes(array, mask, transform=affine)
            # geometry = [
            #     shape(geom)
            #     for geom, _ in it
            # ]
            # binary mask for this class
            mask = (array == id).astype(np.uint8)

            # polygonize the mask itself; filter val==1
            it = rasterio.features.shapes( mask, transform=affine )
            geometry = [
                shape(geom)
                for geom, val in it
                if val == 1
            ]
            append = gpd.GeoDataFrame(dict(
                geometry=geometry,
                feature=label,
            ), crs=4326)
            concat.append(append)

        result = (
            pd.concat(concat)
            .pipe(gpd.GeoDataFrame)
            .to_crs(crs)
            .set_index('feature')
            .pipe(cls.from_frame)
        )
        return result


    def _replace_convexhull(
            self,
            threshold: float | Series
    ) -> Self:
        """
        replace the convex polygons with their envelopes
        Args:
            self: geopandas geodataframe
            threshold: convexity threshold to filter lines

        Returns:
            geopandas geodataframe
        """
        hulls: gpd.GeoSeries = self.frame.convex_hull
        convexity = self.frame.area / hulls.area  # [0, 1]
        logger.debug(f"Applying convexity filter")
        loc = convexity > threshold
        hulls = hulls.loc[loc]
        result: gpd.GeoDataFrame = self.copy()
        result.loc[loc, 'geometry'] = hulls
        logger.debug(f"Replaced {loc.sum()} geometry(ies) with convex hulls")

        return result

    def _fill_holes(self, max_area: ndarray) -> Self:
        logger.debug(f"Starting hole‐filling")
        MAX_AREA = max_area

        max_area = MAX_AREA
        RINGS = shapely.get_rings(self.frame.geometry)
        polygons = shapely.polygons(RINGS)
        area = shapely.area(polygons)
        repeat = self.frame.count_interior_rings() + 1
        indices = np.arange(len(repeat)).repeat(repeat)
        max_area = max_area.repeat(repeat)
        loc = area >= max_area
        msg = (
            f'Filling holes in {len(self)} polygons, '
            f'dropping {np.sum(~loc)} holes out of {len(RINGS)} total holes.'
        )
        logger.debug(msg)
        loc |= (
                pd.Series(indices)
                .groupby(indices, sort=False)
                .cumcount()
                == 0
        )
        rings = RINGS[loc]
        indices = indices[loc]
        data = shapely.polygons(rings, indices=indices)
        result = (
            gpd.GeoSeries(data, index=self.index, crs=self.crs)
            .pipe(self.frame.set_geometry)
            .pipe(self.from_frame, wrapper=self)
        )
        return result

    def postprocess(
            self,
            min_poly_area: Union[float, Series, dict] = None,
            simplify: Union[float, Series, dict] = None,
            max_hole_area: Union[float, Series, dict] = None,
            convexity: Union[float, Series, dict] = None,
            # crs: int = 3857,
    ) -> Self:
        cls = self.__class__
        logger.debug("Starting postprocessing")
        msg = f'Dissolving, simplifying, & exploding {len(self)} (Multi)Polygons'
        if simplify is None:
            simplify = cfg.polygon.simplify
        loc = self.geometry.is_valid
        loc |= self.geometry.isna()
        assert np.all(loc)

        with benchmark(msg):
            result: Self = (
                self.frame
                #
                # .to_crs(crs)
                .dissolve(level='feature', method='coverage')
                # .dissolve(level='feature', method='unary',)
                .set_geometry('geometry')
                .simplify_coverage(simplify)
                .explode()
                .to_frame('geometry')
                # .to_crs(crs)
                .make_valid(
                    method='structure',
                    keep_collapsed=False,
                )
                .explode()
                .to_frame('geometry')
                .pipe(self.__class__.from_frame, wrapper=self)
            )

        RESULT = result

        assert np.all(result.frame.geom_type == 'Polygon')
        if min_poly_area is None:
            min_poly_area = cfg.polygon.min_polygon_area
        if max_hole_area is None:
            max_hole_area = cfg.polygon.max_hole_area
        if convexity is None:
            convexity = cfg.polygon.convexity

        min_poly_area: Union[float, Series]
        simplify: Union[float, Series]
        max_hole_area: Union[float, Series]
        convexity: Union[float, Series]

        # todo: avoid unnecessarily computing area, etc for redundant parameters
        if None is not min_poly_area is not False:
            if isinstance(min_poly_area, dict):
                min_poly_area = (
                    pd.Series(min_poly_area)
                    .reindex(result.index, fill_value=0.0)
                    .values
                )
            elif isinstance(min_poly_area, (float, int)):
                min_poly_area = np.full(len(result), min_poly_area)
            elif isinstance(min_poly_area, pd.Series):
                min_poly_area = (
                    min_poly_area
                    .reindex(result.index, fill_value=0.0)
                    .values
                )
            else:
                raise TypeError(f'Unsupported type for min_poly_area: {type(min_poly_area)}')

            msg = f'Applying area filter'
            logger.debug(msg)
            loc = result.frame.area >= min_poly_area
            result = result.loc[loc]
            msg = f'{len(result)} out of {len(loc)} polygons remaining after area filter'
            logger.debug(msg)

        if None is not max_hole_area is not False:
            if isinstance(max_hole_area, dict):
                max_hole_area = (
                    pd.Series(max_hole_area)
                    .reindex(result.index, fill_value=0.)
                    .values
                )
            elif isinstance(max_hole_area, float):
                max_hole_area = np.full(len(result), max_hole_area)
            elif isinstance(max_hole_area, pd.Series):
                max_hole_area = (
                    max_hole_area
                    .reindex(result.index, fill_value=0.)
                    .values
                )
            if not isinstance(max_hole_area, np.ndarray):
                msg = f'Unsupported type for max_hole_area: {type(max_hole_area)}'
                raise TypeError(msg)

            result = cls._fill_holes(result, max_area=max_hole_area)

        if None is not convexity is not False:
            if isinstance(convexity, dict):
                convexity = (
                    pd.Series(convexity)
                    .reindex(result.index, fill_value=1.0)
                    .values
                )
            elif isinstance(convexity, float):
                convexity = np.full(len(result), convexity)
            elif isinstance(convexity, pd.Series):
                convexity = (
                    convexity
                    .reindex(result.index, fill_value=1.0)
                    .values
                )
            else:
                raise TypeError(f'Unsupported type for convexity: {type(convexity)}')

            result = cls._replace_convexhull(result, threshold=convexity)

        # result = (
        #     cls(result)
        #     .frame
        #     .to_crs(self.crs)
        # )
        result = (
            result.frame
            # .to_crs(self.crs)
            .pipe(self.from_frame, wrapper=self)
        )
        return result

    def plot(
            self,
            *args,
            maxdim: int = 2048,
            background: str | tuple[int, int, int] = 'black',
            simplify: float | None = None,
            show: bool = True,
            **kwargs,
    ) -> PIL.Image.Image:

        # geometry bounds & scale to determine output pixel size
        minx, miny, maxx, maxy = self.geometry.total_bounds
        span_x = max(maxx - minx, 1e-12)
        span_y = max(maxy - miny, 1e-12)
        scale = maxdim / max(span_x, span_y)
        out_w = ceil(span_x * scale)
        out_h = ceil(span_y * scale)
        long_side = max(out_w, out_h)

        # figure canvas sized to raster dims
        dpi = 96
        fig_w_in = out_w / dpi
        fig_h_in = out_h / dpi
        fig, ax = plt.subplots(
            figsize=(fig_w_in, fig_h_in),
            dpi=dpi,
            facecolor=background,
        )
        ax.set_facecolor(background)

        # axis/ticks styled for dark background
        axis_col = 'white'
        labelsize_pt = max(8, int(long_side / dpi * 72 * 0.04))
        ticklen_px = max(4, int(long_side * 0.006))
        ax.tick_params(
            axis='both',
            which='both',
            colors=axis_col,
            direction='out',
            labelsize=labelsize_pt,
            length=ticklen_px,
            width=max(1, ticklen_px // 3),
        )
        for spine in ax.spines.values():
            spine.set_color(axis_col)
            spine.set_linewidth(max(1, ticklen_px // 3))

        # per-feature polygon edge colors and stroke width
        label2color = cfg.label2color
        line_w = kwargs.get('width', max(1, long_side // 1400))

        # iterator yielding Polygon from (Multi)Polygon
        def _iter_polys(g):
            if g.geom_type == 'Polygon':
                yield g
            elif g.geom_type == 'MultiPolygon':
                yield from g.geoms

        # plot each feature polygon (exterior + holes)
        it = self.frame.groupby('feature', observed=False).geometry
        for feature, series in it:
            colour = label2color.get(feature, axis_col)

            for geom in series:
                if simplify:
                    geom = geom.simplify(simplify, preserve_topology=True)

                for poly in _iter_polys(geom):
                    if poly.is_empty:
                        continue

                    # exterior ring
                    ext = poly.exterior
                    if ext is not None:
                        ext_xy = np.asarray(ext.coords)
                        if ext_xy.shape[0] >= 2:
                            ax.plot(
                                ext_xy[:, 0],
                                ext_xy[:, 1],
                                color=colour,
                                linewidth=line_w,
                                solid_joinstyle='round',
                                solid_capstyle='round',
                                zorder=3,
                            )

                    # interior rings (holes)
                    for ring in poly.interiors:
                        ring_xy = np.asarray(ring.coords)
                        if ring_xy.shape[0] >= 2:
                            ax.plot(
                                ring_xy[:, 0],
                                ring_xy[:, 1],
                                color=colour,
                                linewidth=max(1, line_w // 2),
                                solid_joinstyle='round',
                                solid_capstyle='round',
                                zorder=3,
                            )

        # configure map frame
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.98)

        # rasterise the figure to a PIL image
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format='png',
            facecolor=fig.get_facecolor(),
            bbox_inches=None,
            pad_inches=0.0,
        )
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')

        # optionally show live window
        if show:
            try:
                fig.canvas.manager.set_window_title('tile2net — mask2poly')
            except Exception:
                pass
            plt.show(block=False)
            plt.pause(0.001)
        else:
            plt.close(fig)

        return pil_img
