from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import *

import imageio.v2
import imageio.v3
import math
import numpy as np
import pandas as pd
import pyproj
import shapely
from sympy.codegen.fnodes import intent_in
from tqdm.auto import tqdm

from tile2net.raster import util
from tile2net.tiles import util
from tile2net.tiles.cfg.logger import logger
from tile2net.tiles.explore import explore
from tile2net.tiles.fixed import GeoDataFrameFixed
from . import tile
from .colormap import ColorMap
from .corners import Corners
from .stitcher import Stitcher
from .static import Static
from .tile import Tile

if False:
    import folium
    from ..intiles import InTiles
    from ..segtiles import SegTiles
    from ..vectiles import VecTiles


class Tiles(
    GeoDataFrameFixed,
):

    @property
    def lonmax(self) -> pd.Series:
        return self['lonmax']

    @property
    def lonmin(self) -> pd.Series:
        return self['lonmin']

    @property
    def latmax(self) -> pd.Series:
        return self['latmax']

    @property
    def latmin(self) -> pd.Series:
        return self['latmin']

    @tile.cached_property
    def intiles(self) -> InTiles:
        ...

    @tile.cached_property
    def instance(self) -> InTiles:
        ...

    @property
    def segtiles(self) -> SegTiles:
        return self.intiles.segtiles

    @property
    def vectiles(self) -> VecTiles:
        return self.intiles.vectiles

    @property
    def xtile(self) -> pd.Index:
        """Tile integer X"""
        try:
            return self.index.get_level_values('xtile')
        except KeyError:
            return self['xtile']

    @property
    def ytile(self) -> pd.Index:
        """Tile integer Y"""
        try:
            return self.index.get_level_values('ytile')
        except KeyError:
            return self['ytile']

    @Tile
    def tile(self):
        """ Wrapper for tile attributes, such as tile scale. """
        # This code block is just semantic sugar and does not run.
        # This is how one would set the attributes:
        self.tile.scale = ...
        self.tile.zoom = ...
        self.tile.dimension = ...

    @classmethod
    def from_integers(
            cls,
            xtile: pd.Series | pd.Index | np.ndarray,
            ytile: pd.Series | pd.Index | np.ndarray,
            scale: int,
    ) -> Self:
        """
        Construct Tiles from integer tile numbers.
        This allows for a non-rectangular grid of tiles.
        """
        if isinstance(ytile, (pd.Series, pd.Index)):
            tn = ytile.values
        elif isinstance(ytile, np.ndarray):
            tn = ytile
        else:
            raise TypeError('ytile must be a Series, Index, or ndarray')
        tn = tn.astype('uint32')

        if isinstance(xtile, (pd.Series, pd.Index)):
            tw = xtile.values
        elif isinstance(xtile, np.ndarray):
            tw = xtile
        else:
            raise TypeError('xtile must be a Series, Index, or ndarray')
        tw = tw.astype('uint32')

        te = tw + 1
        ts = tn + 1
        names = 'xtile ytile'.split()
        index = pd.MultiIndex.from_arrays([xtile, ytile], names=names)
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

        result = cls(
            data=data,
            geometry=geometry,
            index=index,
            # crs=4326,
            crs=3857,
        )
        result.tile.scale = scale
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
            tiles: Tiles,
            scale: int,
            fill: bool = True,
    ) -> Self:
        """
        Rescale tiles to a new scale.
        If the new scale is larger, the tiles are filled with zeros.
        If the new scale is smaller, the tiles are downscaled.
        """
        scaled = tiles.to_scale(scale, fill=fill)
        xtile = scaled.xtile
        ytile = scaled.ytile
        result = cls.from_integers(
            xtile=xtile,
            ytile=ytile,
            scale=scale,
        )
        assert result.tile.scale == scale
        return result

    @cached_property
    def r(self) -> pd.Series:
        """Row of the tile within the overall grid."""
        result = (
            self.ytile
            .to_series()
            .set_axis(self.index)
            .sub(self.ytile.min())
        )
        return result

    @cached_property
    def c(self) -> pd.Series:
        """Column of the tile within the overall grid."""
        result = (
            self.xtile
            .to_series()
            .set_axis(self.index)
            .sub(self.xtile.min())
        )
        return result

    @property
    def name(self) -> str:
        return self.cfg.name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise TypeError('Tiles.name must be a string')
        self.cfg.name = value

    def explore(
            self,
            *args,
            loc=None,
            tile_color='grey',
            subset_color='yellow',
            tiles: str = 'cartodbdark_matter',
            m=None,
            **kwargs
    ) -> folium.map:
        import folium
        if loc is None:
            m = explore(
                self,
                color=tile_color,
                name='lines',
                *args,
                **kwargs,
                tiles=tiles,
                m=m,
                style_kwds=dict(
                    # fill=False,
                    fillOpacity=0,
                )
            )
        else:
            tiles = self.loc[loc]
            loc = ~self.index.isin(tiles.index)
            m = explore(
                self.loc[loc],
                color=tile_color,
                name='tiles',
                *args,
                **kwargs,
                tiles=tiles,
                m=m,
                style_kwds=dict(
                    fill=False,
                    fillOpacity=0,
                )
            )
            m = explore(
                tiles,
                color=subset_color,
                name='subset',
                *args,
                **kwargs,
                tiles=tiles,
                m=m,
                style_kwds=dict(
                    fill=False,
                    fillOpacity=0,
                )
            )

        folium.LayerControl().add_to(m)
        return m

    @ColorMap
    def colormap(self):
        # This code block is just semantic sugar and does not run.
        # This allows us to apply colormaps to tensors, ndarrays, and images.
        # todo: allow setting custom colormaps
        # See:
        self.colormap.__call__(...)
        self.colormap(...)

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        if (
                args
                and callable(args[0])
        ):
            super().__init__(*args[1:], **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(
            self,
            instance: Tiles,
            value: type[Tiles],
    ):
        value.__name__ = self.__name__
        copy = value.copy()
        # problem is, when we set like this, we lose all attrs like tile.scale
        # copy.attrs.clear()
        instance.attrs[self.__name__] = copy

    @Static
    def static(self):
        # This code block is just semantic sugar and does not run.
        # This is a namespace container for static files:
        self.static.hrnet_checkpoint = ...
        self.static.snapshot = ...

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

    def to_scale(
            self,
            scale: int,
            fill: bool = True,
    ) -> Self:
        """
        scale:
            new scale of tiles
        fill:
            if True, fills missing tiles when making larger tiles.
            else, the larger tiles that have missing tiles are dropped.
        """

        mosaic_length = 2 ** abs(self.tile.scale - scale)
        if self.tile.scale < scale:
            # into smaller tiles
            smaller = self.index.to_frame()
            topleft = smaller.mul(mosaic_length)
            arange = np.arange(mosaic_length)
            x, y = np.meshgrid(arange, arange, indexing='ij')
            txy: np.ndarray = (
                np.stack((x, y), -1)
                .reshape(-1, 2)
                .__add__(topleft.values[:, None, :])
                .reshape(-1, 2)
            )
            xtile, ytile = txy.T
            result = self.from_integers(
                xtile,
                ytile,
                scale
            )
            assert len(result) == len(self) * (mosaic_length ** 2)
            assert not result.index.duplicated().any()

        elif self.tile.scale > scale:
            # into larger tiles
            frame: Self = (
                self.index
                .to_frame()
                .floordiv(mosaic_length)
            )
            if fill:
                frame = frame.drop_duplicates()
            else:
                loc = (
                    frame
                    .drop_duplicates()
                    .pipe(pd.MultiIndex.from_frame)
                )
                loc = (
                    frame
                    .groupby(frame.columns.tolist(), sort=False)
                    .size()
                    .eq(mosaic_length ** 2)
                    .loc[loc]
                )
                frame = frame.loc[loc]

            result = self.from_integers(
                frame.xtile,
                frame.ytile,
                scale
            )

            assert len(self) > len(result)
            assert not result.index.duplicated().any()

        else:
            # same scale
            result = self.copy()

        result.attrs.update(self.attrs)
        result.tile.scale = scale
        return result

    def _to_scale(
            self,
            dimension: int = None,
            length: int = None,
            mosaic: int = None,
            scale: int = None,
    ) -> int:

        n = sum(
            arg is not None
            for arg in (dimension, mosaic, scale, length)
        )
        if n != 1:
            msg = (
                'You must specify exactly one of dimension, length, mosaic, or scale '
                'to set the inference tiles.'
            )
            raise ValueError(msg)

        """get scale from dimension, length, or mosaic"""
        if dimension:

            if (
                    not isinstance(dimension, int)
                    or dimension <= 0
                    or (dimension & (dimension - 1)) != 0
            ):
                raise ValueError('Dimension must be a positive power of 2.')
            dscale = int(math.log2(dimension / self.tile.dimension))
            scale = self.tile.scale + dscale

        elif length:
            if (
                    not isinstance(length, int)
                    or length <= 0
                    or (length & (length - 1)) != 0
            ):
                raise ValueError('Length must be a positive power of 2.')
            scale = self.tile.scale - int(math.log2(length))

        elif mosaic:
            if (
                    not isinstance(mosaic, int)
                    or mosaic <= 0
                    or (mosaic & (mosaic - 1)) != 0
            ):
                raise ValueError('Mosaic must be a positive power of 2.')
            marea = int(math.log2(mosaic))
            dscale = int(math.sqrt(marea))
            scale = self.tile.scale - dscale

        else:
            msg = 'You must specify either dimension, length, or mosaic to set the scale.'
            raise ValueError(msg)

        return scale

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

    def _stitch(
        self,
        small_files: pd.Series,
        big_files: pd.Series,
        small_tiles: Tiles,
        big_tiles: Tiles,
        r: pd.Series,
        c: pd.Series,
        background: int = 0,
        force=False
    ):


        if not force:
            loc = ~big_files.map(os.path.exists)
            small_files = small_files.loc[loc]
            row = r.loc[loc]
            col = c.loc[loc]
            big_files: pd.Series = big_files.loc[loc]

        stitched = big_files.drop_duplicates()
        n_missing = len(small_files)
        n_total = len(stitched)
        if n_missing == 0:  # nothing to do
            msg = f'All {n_total:,} mosaics are already stitched.'
            logger.info(msg)
            # return
        else:
            msg = (
                f'Stitching {n_missing:,} '
                f'{small_tiles.__name__}.{small_files.name} '
                f'into {n_total:,}'
                f'{big_tiles.__name__}.{big_files.name}'
            )
            logger.info(msg)


        loader = Stitcher(
            infiles=small_files,
            row=row,
            col=col,
            tile_shape=small_tiles.tile.shape,
            mosaic_shape=big_tiles.tile.shape,
            outfiles=big_files,
            background=background,
        )

        loader.run(max_workers=os.cpu_count())
        msg = 'Not all stitched mosaics were written to disk.'
        assert big_files.map(os.path.exists).all(), msg
