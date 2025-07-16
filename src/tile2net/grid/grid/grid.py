from __future__ import annotations

import numpy as np
import pandas as pd

from tile2net.raster import util
from tile2net.tiles import util

from .. import util, static
import shapely
import pyproj
from pandas import MultiIndex
from ..framewrapper import FrameWrapper
from .. import frame

import copy
from typing import *

from functools import *

from geopandas import GeoDataFrame
from pandas import Series, Index

if False:
    from ..seggrid.seggrid import SegGrid
    from ..ingrid.ingrid import InGrid
    from ..vecgrid.vecgrid import VecGrid


class Grid(FrameWrapper):

    @property
    def tiles(self) -> GeoDataFrame:
        return self.frame

    @frame.column
    def lonmax(self) -> Series:
        """
        accessor for self.tiles.lonmax
        """

    @frame.column
    def lonmin(self):
        """
        accessor for self.tiles.lonmin
        """

    @frame.column
    def latmax(self) -> Series:
        """
        accessor for self.tiles.latmax
        """

    @frame.column
    def latmin(self) -> Series:
        """
        accessor for self.tiles.latmin
        """

    @frame.index
    def xtile(self):
        """
        accessor for self.tiles.index.get_level_values('xtile')
        """

    @frame.index
    def ytile(self) -> Index | Series:
        """
        accessor for self.tiles.index.get_level_values('ytile')
        """

    @cached_property
    def scale(self) -> int:
        """
        Tile scale; the XYZ scale of the tiles.
        Higher value means smaller area.
        """

    @cached_property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dimension, self.dimension, 3

    @cached_property
    def length(self) -> int:
        """How many input tiles comprise a tile of this class"""
        raise NotImplemented

    @cached_property
    def area(self):
        return self.length ** 2

    @cached_property
    def r(self) -> Series:
        """
        Row of the tile within the overall grid.
        """
        result = (
            self.ytile
            .to_series()
            .set_axis(self.tiles.index)
            .sub(self.ytile.min())
        )
        return result

    @cached_property
    def c(self) -> Series:
        """
        Column of the tile within the overall grid.
        """
        result = (
            self.xtile
            .to_series()
            .set_axis(self.tiles.index)
            .sub(self.xtile.min())
        )
        return result

    tiles: GeoDataFrame
    @cached_property
    def ingrid(self) -> InGrid:
        ...

    @property
    def seggrid(self) -> SegGrid:
        return self.ingrid.seggrid

    @property
    def vecgrid(self) -> VecGrid:
        return self.ingrid.vecgrid

    @classmethod
    def from_integers(
            cls,
            xtile: Series | Index | np.ndarray,
            ytile: Series | Index | np.ndarray,
            scale: int,
    ) -> Self:
        """
        Construct Tiles from integer tile numbers.
        This allows for a non-rectangular grid of tiles.
        """
        if isinstance(ytile, (Series, Index)):
            tn = ytile.values
        elif isinstance(ytile, np.ndarray):
            tn = ytile
        else:
            raise TypeError('ytile must be a Series, Index, or ndarray')
        tn = tn.astype('uint32')

        if isinstance(xtile, (Series, Index)):
            tw = xtile.values
        elif isinstance(xtile, np.ndarray):
            tw = xtile
        else:
            raise TypeError('xtile must be a Series, Index, or ndarray')
        tw = tw.astype('uint32')

        te = tw + 1
        ts = tn + 1
        names = 'xtile ytile'.split()
        index = MultiIndex.from_arrays([xtile, ytile], names=names)
        gw, gn = util.num2deg(tw, tn, zoom=scale)
        ge, gs = util.num2deg(te, ts, zoom=scale)
        trans = (
            pyproj.proj.Transformer
            .from_crs(4326, 3857, always_xy=True)
            .transform
        )
        pw, pn = trans(gw, gn)
        pe, ps = trans(ge, gs)
        geometry = shapely.box(pw, pn, pe, ps)

        data = dict(
            lonmin=gw,
            latmax=gn,
            lonmax=ge,
            latmin=gs,
        )
        tiles = GeoDataFrame(
            data=data,
            index=index,
            geometry=geometry,
            crs=3857
        )
        result = cls(tiles=tiles, scale=scale)
        return result

    @classmethod
    def from_ranges(
            cls,
            xmin: pd.Series | pd.Index | np.ndarray,
            ymin: pd.Series | pd.Index | np.ndarray,
            xmax: pd.Series | pd.Index | np.ndarray,
            ymax: pd.Series | pd.Index | np.ndarray,
            scale: int,
    ) -> Self:
        """
        Construct Tiles from ranges of tile numbers.
        Generates every (xtile, ytile) pair within each [xmin‥xmax] × [ymin‥ymax] extent
        and delegates to `from_integers`.
        """
        for name, arr in (
                ('xmin', xmin),
                ('ymin', ymin),
                ('xmax', xmax),
                ('ymax', ymax),
        ):
            if not isinstance(arr, (pd.Series, pd.Index, np.ndarray)):
                raise TypeError(f'{name} must be a Series, Index, or ndarray')

        xmin_arr = np.asarray(xmin, dtype='uint32')
        ymin_arr = np.asarray(ymin, dtype='uint32')
        xmax_arr = np.asarray(xmax, dtype='uint32')
        ymax_arr = np.asarray(ymax, dtype='uint32')

        if not (xmin_arr.shape == ymin_arr.shape == xmax_arr.shape == ymax_arr.shape):
            raise ValueError('xmin, ymin, xmax, ymax must have identical shapes')

        if (xmin_arr > xmax_arr).any():
            raise ValueError('xmin must be ≤ xmax element-wise')
        if (ymin_arr > ymax_arr).any():
            raise ValueError('ymin must be ≤ ymax element-wise')

        xtiles: list[np.ndarray] = []
        ytiles: list[np.ndarray] = []

        for x0, y0, x1, y1 in zip(xmin_arr, ymin_arr, xmax_arr, ymax_arr):
            xs = np.arange(x0, x1, dtype='uint32')
            ys = np.arange(y0, y1, dtype='uint32')
            gx, gy = np.meshgrid(xs, ys)  # shape (len(ys), len(xs))
            xtiles.append(gx.ravel())
            ytiles.append(gy.ravel())

        xtile_all = np.concatenate(xtiles)
        ytile_all = np.concatenate(ytiles)

        return cls.from_integers(
            xtile=xtile_all,
            ytile=ytile_all,
            scale=scale,
        )

    @classmethod
    def from_rescale(
            cls,
            grid: Self,
            # tiles: Tiles,
            scale: int,
            fill: bool = True,
    ) -> Self:
        """
        Rescale tiles to a new scale.
        If the new scale is larger, the tiles are filled with zeros.
        If the new scale is smaller, the tiles are downscaled.
        """

        # scaled = tiles.to_scale(scale, fill=fill)
        # xtile = scaled.xtile
        # ytile = scaled.ytile
        # result = cls.from_integers(
        #     xtile=xtile,
        #     ytile=ytile,
        #     scale=scale,
        # )
        # assert result.tile.scale == scale
        # return result
        #

    def to_padding(self, pad: int = 1) -> Self:
        """ Pad each tile by `pad` tiles in each direction. """
        padded = (
            self
            .to_corners(self.tile.scale)
            .to_padding(pad)
            .to_tiles()
            .pipe(self.__class__)
            .sort_index()
        )
        assert self.xtile.min() - pad == padded.xtile.min()
        assert self.ytile.min() - pad == padded.ytile.min()
        assert self.xtile.max() + pad == padded.xtile.max()
        assert self.ytile.max() + pad == padded.ytile.max()
        if pad >= 0:
            assert self.index.isin(padded.index).all()
        padded.attrs.update(self.attrs)
        return padded

    def to_corners(self, scale: int = None) -> Corners:
        if scale is None:
            scale = self.tile.scale

        length = 2 ** (self.tile.scale - scale)
        result = Corners.from_data(
            xmin=self.xtile.values // length,
            ymin=self.ytile.values // length,
            xmax=(self.xtile.values + 1) // length,
            ymax=(self.ytile.values + 1) // length,
            scale=scale,
            index=self.index,
        )
        return result


    frame: GeoDataFrame