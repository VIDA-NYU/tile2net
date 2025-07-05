from __future__ import annotations

import os
from pathlib import Path
from typing import *

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import rasterio.features
import shapely
import shapely.wkt
import skimage
from affine import Affine
from numpy import ndarray
from pandas import Series
from shapely.geometry import shape
import warnings
from PIL import Image
import skimage.io  # assume already in deps

from tile2net.raster.tile_utils.geodata_utils import _check_skimage_im_load
from tile2net.tiles.cfg.logger import logger
from ..benchmark import benchmark
from ..cfg import cfg
from ..fixed import GeoDataFrameFixed

os.environ['USE_PYGEOS'] = '0'


class Mask2Poly(
    GeoDataFrameFixed,
):


    @classmethod
    def from_path(
            cls,
            path: str | Path,
            affine: Affine,
    ) -> Self:

        original_max = Image.MAX_IMAGE_PIXELS

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', Image.DecompressionBombWarning)
            Image.MAX_IMAGE_PIXELS = None           # avoid the error variant
            array = skimage.io.imread(str(path))

        Image.MAX_IMAGE_PIXELS = original_max       # restore global state
        return cls.from_array(array, affine)


    @classmethod
    def from_parquets(cls, files: Iterable[str | Path], ) -> Self:
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

    @classmethod
    def from_array(
            cls,
            array: np.ndarray,
            affine: Affine,
            crs=4326,
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
            mask = np.array(array == id, dtype=np.uint8)
            it = rasterio.features.shapes(array, mask, transform=affine)
            geometry = [
                shape(geom)
                for geom, _ in it
            ]
            append = gpd.GeoDataFrame(dict(
                geometry=geometry,
                feature=label,
            ), crs=crs)
            concat.append(append)

        result = (
            pd.concat(concat)
            .pipe(cls)
            .set_index('feature')
        )


        return result

    @classmethod
    def mask_to_poly_geojson(
            cls,
            source,
            transform: Affine,
    ) -> Self:
        it = (
            (shape(geom), val)
            for geom, val in
            rasterio.features.shapes(source, transform=transform)
        )
        columns = 'geometry value'.split()
        result = gpd.GeoDataFrame.from_records(it, columns=columns)

        return result

    @classmethod
    def preds_to_binary(
            cls,
            pred_arr: Union[np.ndarray, str],
            channel_scaling=None,
            bg_threshold=0
    ) -> np.ndarray:
        """
        *Adopted from the Solaris library to overcome dependency issues*

        Convert a set of predictions from a neural net to a binary mask.

        Parameters
        ----------
        pred_arr : :class:`numpy.ndarray`
            A set of predictions generated by a neural net (generally in ``float``
            dtype). This can be a 2D array or a 3D array, in which case it will
            be convered to a 2D mask output with optional channel scaling (see
            the `channel_scaling` argument). If a filename is provided instead of
            an array, the image will be loaded using scikit-image.
        channel_scaling : `list`-like of `float`s, optional
            If `pred_arr` is a 3D array, this argument defines how each channel
            will be combined to generate a binary output. channel_scaling should
            be a `list`-like of length equal to the number of channels in
            `pred_arr`. The following operation will be performed to convert the
            multi-channel prediction to a 2D output ::

                sum(pred_arr[channel]*channel_scaling[channel])

            If not provided, no scaling will be performend and channels will be
            summed.

        bg_threshold : `int` or `float`, optional
            The cutoff to set to distinguish between background and foreground
            pixels in the final binary mask. Binarization takes place *after*
            channel scaling and summation (if applicable). Defaults to 0.

        Returns
        -------
        mask_arr : :class:`numpy.ndarray`
            A 2D boolean ``numpy`` array with ``True`` for foreground pixels and
            ``False`` for background.
        """
        pred_arr = _check_skimage_im_load(pred_arr).copy()

        mask_arr = (
                pred_arr
                .__gt__(bg_threshold)
                .astype('uint8')
                * 255
        )
        return mask_arr

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
        hulls: gpd.GeoSeries = self.convex_hull
        convexity = self.area / hulls.area  # [0, 1]
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
        RINGS = shapely.get_rings(self.geometry)
        polygons = shapely.polygons(RINGS)
        area = shapely.area(polygons)
        repeat = shapely.get_num_interior_rings(self.geometry) + 1
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
            .pipe(self.set_geometry)
        )
        return result

    def postprocess(
            self,
            min_poly_area: Union[float, Series, dict] = None,
            simplify: Union[float, Series, dict] = None,
            grid_size: Union[float, Series, dict] = None,
            max_hole_area: Union[float, Series, dict] = None,
            convexity: Union[float, Series, dict] = None,
            crs: int = 3857,
    ) -> Self:
        cls = self.__class__
        logger.debug("Starting postprocessing")
        msg = f'Dissolving & exploding {len(self)} polygon(s)'
        if simplify is None:
            simplify = cfg.polygon.simplify
        with benchmark(msg):
            result: Self = (
                self
                .to_crs(crs)
                .dissolve(level='feature')
                .set_geometry('geometry')
                .simplify_coverage(simplify)
                .explode()
                .to_frame('geometry')
                .pipe(self.__class__)
            )
            assert np.all(result.is_valid)

        logger.debug(f"Dissolved & exploded: {len(result)} polygon(s)")

        RESULT = result

        if min_poly_area is None:
            min_poly_area = cfg.polygon.min_polygon_area
        # if grid_size is None:
        #     grid_size = cfg.polygon.grid_size
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
            loc = result.area >= min_poly_area
            result = result.loc[loc]
            msg = f'{len(result)} out of {len(loc)} polygons remaining after area filter'
            logger.debug(msg)

        # if None is not grid_size is not False:
        #     result = result.set_precision(grid_size=grid_size)


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
                raise TypeError(
                    f'Unsupported type for max_hole_area: {type(max_hole_area)}'
                )

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

        result = cls(result)
        return result

