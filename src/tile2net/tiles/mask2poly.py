from __future__ import annotations

from typing import *

import os
from pathlib import Path
from typing import *
from typing import TypedDict

import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely.wkt
import skimage
from affine import Affine
from rasterio import features
from typing_extensions import Unpack  # <3.12

from tile2net.raster.tile_utils.geodata_utils import _reduce_geom_precision, list_to_affine, _check_skimage_im_load
from .fixed import GeoDataFrameFixed

pd.options.mode.chained_assignment = None
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import shapely
from functools import reduce
from tile2net.raster.tile_utils.momepy_shapes import *


class ChannelSpec(TypedDict):
    channel: int
    hole: int | None


class Mask2Poly(
    GeoDataFrameFixed,
):
    @classmethod
    def from_path(
            cls,
            path: str | Path,
            affine: Affine,
            crs: str | pyproj.CRS = "EPSG:3857",
            postprocess=True,
            **classes: Unpack[dict[str, ChannelSpec]]
    ) -> Self:
        array = skimage.io.imread(str(path))
        return cls.from_array(array, affine, crs=crs)

    @classmethod
    def from_array(
            cls,
            array: np.ndarray,
            affine: Affine,
            crs: str | pyproj.CRS = "EPSG:3857",
            postprocess=True,
            **classes: Unpack[dict[str, ChannelSpec]]
    ) -> Self:
        ARRAY = array
        concat: list[gpd.GeoDataFrame] = []
        # todo: support postprocesing
        for ftype, params in classes.items():
            hole = params.get('hole', None)
            channel = params['channel']
            array = ARRAY[:, :, channel]
            if not array.max() > 0:
                continue
            geoms_class = cls.mask_to_poly_geojson(array)

            if postprocess:
                result = geoms_class.postprocess()
            else:
                result = geoms_class

        return

    @classmethod
    def mask_to_poly_geojson(
            cls,
            pred_arr,
            channel_scaling=None,
            reference_im=None,
            min_area=20,
            bg_threshold=0,
            do_transform=None,
    ) -> Self:
        """
        *Adopted from the Solaris library to overcome dependency issues*

        Get polygons from an image mask.

        Parameters
        ----------
        pred_arr : :class:`numpy.ndarray`
            A 2D array of integers. Multi-channel masks are not supported, and must
            be simplified before passing to this function. Can also pass an image
            file path here.
        channel_scaling : :class:`list`-like, optional
            If `pred_arr` is a 3D array, this argument defines how each channel
            will be combined to generate a binary output. channel_scaling should
            be a `list`-like of length equal to the number of channels in
            `pred_arr`. The following operation will be performed to convert the
            multi-channel prediction to a 2D output ::

                sum(pred_arr[channel]*channel_scaling[channel])

            If not provided, no scaling will be performend and channels will be
            summed.
        reference_im : str, optional
            The path to a reference geotiff to use for georeferencing the polygons
            in the mask. Required if saving to a GeoJSON (see the ``output_type``
            argument), otherwise only required if ``do_transform=True``.
        output_path : str, optional
            Path to save the output file to. If not provided, no file is saved.
        output_type : ``'csv'`` or ``'geojson'``, optional
            If ``output_path`` is provided, this argument defines what type of file
            will be generated - a CSV (``output_type='csv'``) or a geojson
            (``output_type='geojson'``).
        min_area : int, optional
            The minimum area of a polygon to retain. Filtering is done AFTER
            any coordinate transformation, and therefore will be in destination
            units.
        bg_threshold : int, optional
            The cutoff in ``mask_arr`` that denotes background (non-object).
            Defaults to ``0``.
        simplify : bool, optional
            If ``True``, will use the Douglas-Peucker algorithm to simplify edges,
            saving memory and processing time later. Defaults to ``False``.
        tolerance : float, optional
            The tolerance value to use for simplification with the Douglas-Peucker
            algorithm. Defaults to ``0.5``. Only has an effect if
            ``simplify=True``.

        Returns
        -------
        gdf : :class:`geopandas.GeoDataFrame`
            A :class:`GeoDataFrame` of polygons.

        """
        mask_arr = cls.preds_to_binary(pred_arr, channel_scaling, bg_threshold)

        if do_transform and reference_im is None:
            raise ValueError(
                'Coordinate transformation requires a reference image.')

        if do_transform:
            with rasterio.open(reference_im) as ref:
                transform = ref.transform
                crs = ref.crs
                ref.close()
        else:
            transform = Affine(1, 0, 0, 0, 1, 0)  # identity transform
            crs = rasterio.crs.CRS()

        mask = mask_arr > bg_threshold
        mask = mask.astype('uint8')

        polygon_generator = features.shapes(
            mask_arr,
            transform=transform,
            mask=mask
        )
        data = polygon_generator
        columns = 'geometry value'.split()
        result = (
            cls
            .from_records(data, columns=columns)
            .set_crs(crs)
        )

        return result


    def convert_poly_coords(
            self,
            affine_obj: Affine,
            inverse=False,
            precision=None
    ):
        """
        *Adopted from the Solaris library to overcome dependency issues*

        Georegister geometry objects currently in pixel coords or vice versa.

        Parameters
        ----------
        self : :class:`shapely.geometry.shape` or str
            A :class:`shapely.geometry.shape`, or WKT string-formatted geometry
            object currently in pixel coordinates.
        affine_obj: list or :class:`affine.Affine`
            An affine transformation to apply to `geom` in the form of an
            ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
            Required if not using `raster_src`.
        inverse : bool, optional
            If true, will perform the inverse affine transformation, going from
            geospatial coordinates to pixel coordinates.
        precision : int, optional
            Decimal precision for the polygon output. If not provided, rounding
            is skipped.

        Returns
        -------
        out_geom
            A geometry in the same format as the input with its coordinate system
            transformed to match the destination object.
        """

        if not affine_obj:
            raise ValueError("affine_obj must be provided.")

        if isinstance(affine_obj, Affine):
            affine_xform = affine_obj
        else:
            # assume it's a list in either gdal or "standard" order
            if len(affine_obj) == 9:  # if it's straight from rasterio
                affine_obj = affine_obj[0:6]
            affine_xform = list_to_affine(affine_obj)

        if inverse:  # geo->px transform
            affine_xform = ~affine_xform

        if isinstance(self, str):
            # get the polygon out of the wkt string
            g = shapely.wkt.loads(self)
        elif isinstance(self, shapely.geometry.base.BaseGeometry):
            g = self
        else:
            raise TypeError(
                'The provided geometry is not an accepted format. '
                'This function can only accept WKT strings and '
                'shapely geometries.'
            )

        matrix = [
            affine_xform.a,
            affine_xform.b,
            affine_xform.d,
            affine_xform.e,
            affine_xform.xoff,
            affine_xform.yoff
        ]
        xformed_g = shapely.affinity.affine_transform(g, matrix)
        if isinstance(self, str):
            # restore to wkt string format
            xformed_g = shapely.wkt.dumps(xformed_g)
        if precision is not None:
            xformed_g = _reduce_geom_precision(xformed_g, precision=precision)

        return xformed_g

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

        if len(pred_arr.shape) == 3:
            if pred_arr.shape[0] < pred_arr.shape[-1]:
                pred_arr = np.moveaxis(pred_arr, 0, -1)
            if channel_scaling is None:  # if scale values weren't provided
                channel_scaling = np.ones(
                    shape=(pred_arr.shape[-1]),
                    dtype='float'
                )
            pred_arr = np.sum(pred_arr * np.array(channel_scaling), axis=-1)

        mask_arr = (
                pred_arr
                .__gt__(bg_threshold)
                .astype('uint8')
                * 255
        )
        return mask_arr

    def _sanitize(
            self,
            simplify: float = 0.8,
            min_area: float = 20,
    ) -> Self:
        result = (
            self
            .buffer(0.)
            .make_valid()
        )
        if min_area:
            loc = result.area >= min_area
            result = result[loc]
        if simplify:
            result = result.simplify(tolerance=simplify)
        result = self.__class__(result)
        return result

    def _replace_convexhull(
            self,
            convex=0.8
    ) -> Self:
        """
        replace the convex polygons with their envelopes
        Args:
            self: geopandas geodataframe
            convex:: convexity threshold to filter lines

        Returns:
            geopandas geodataframe
        """
        hulls: gpd.GeoSeries = self.convex_hull
        convexity = self.area / hulls.area
        loc = convexity > convex
        hulls = hulls.loc[loc]
        result: gpd.GeoDataFrame = self.copy()
        result.update(hulls)
        return result

    def _fill_holes(
            self,
            max_area: float,
    ) -> Self:
        """
        finds holes in the polygons
        Parameters
        ----------
        gs: gpd.GeoSeries
            the GeoSeries of Shapely Polygons to be filled
        max_area: int
            maximum area of holes to be filled

        Returns
        -------
        newgeom: list[shapely.geometry.Polygon]
            list of polygons with holes filled
        """
        RINGS = shapely.get_rings(self)
        area = shapely.area(RINGS)
        repeat = shapely.get_num_interior_rings(self) + 1
        indices = np.arange(len(repeat)).repeat(repeat)
        loc = area >= max_area
        loc |= (
                pd.Series(indices)
                .groupby(indices)
                .cumcount()
                == 0
        )
        rings = RINGS[loc]
        indices = indices[loc]
        data = shapely.polygons(rings, indices=indices)
        index = self.index[loc]
        geometry = gpd.GeoSeries(data, index=index, crs=self.crs)
        result = self.set_geometry(geometry).pipe(self.__class__)
        return result

    @classmethod
    def postprocess(
            self
    ) -> gpd.GeoDataFrame:
        ...

    # Mask2Poly.from_array(
    #     road=dict(
    #         channel=1,
    #         hole=None,
    #     ),
    #     crosswalk=dict(
    #         channel=0,
    #         hole=15,
    #     ),
    #     sidewalk=dict(
    #         channel=2,
    #         hole=None
    #     ),
    # )


"""
1. sanitize
geoms_class['geometry'] = geoms_class['geometry'].apply(cls.convert_poly_coords, affine_obj=affine)
loc = geoms_class.geometry.notna()
loc &= geoms_class.is_valid
geoms_class: gpd.GeoDataFrame = geoms_class[loc]

todo: tile2tfm
groupby-apply 

if hole is not None:
    geoms_class_met = (
        geoms_class
        .to_crs(3857)
        .explode()
        .reset_index(drop=True)
    )
    loc = geoms_class_met.notna()
    geoms_class_filtered = geoms_class_met.loc[loc]
    geoms_class_filtered["geometry"] = geoms_class_filtered.apply(
        cls._fill_holes,
        args=(class_hole_size,),
        axis=1
    )
    simplified = cls._replace_convexhull(geoms_class_filtered)
    # return simplified
    
"""
