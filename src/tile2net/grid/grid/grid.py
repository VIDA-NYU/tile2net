from __future__ import annotations
from tile2net.tiles.cfg.logger import logger

import threading
from functools import *
from pathlib import Path
from typing import *

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pyproj
import shapely
from geopandas import GeoDataFrame
from pandas import MultiIndex
from pandas import Series, Index

from tile2net.raster import util
from tile2net.grid import util
from .corners import Corners
from .. import frame
from .. import util
from ..framewrapper import FrameWrapper
from .file import File
from ..cfg import cfg

if False:
    from ..seggrid.seggrid import SegGrid
    from ..ingrid.ingrid import InGrid
    from ..vecgrid.vecgrid import VecGrid

tls = threading.local()

class Grid(FrameWrapper):
    @File
    def file(self):
        ...

    @frame.column
    def lonmax(self) -> Series:
        """
        accessor for self.frame.lonmax
        """

    @frame.column
    def lonmin(self):
        """
        accessor for self.frame.lonmin
        """

    @frame.column
    def latmax(self) -> Series:
        """
        accessor for self.frame.latmax
        """

    @frame.column
    def latmin(self) -> Series:
        """
        accessor for self.frame.latmin
        """

    @frame.index
    def xtile(self):
        """
        accessor for self.frame.index.get_level_values('xtile')
        """

    @frame.index
    def ytile(self) -> Index | Series:
        """
        accessor for self.frame.index.get_level_values('ytile')
        """

    @cached_property
    def scale(self) -> int:
        """
        Tile scale; the XYZ scale of the tiles.
        Higher value means smaller area.
        """

    @cached_property
    def zoom(self) -> int:
        """
        Zoom level of the tile.
        """
        return self.scale

    @cached_property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        try:
            # noinspection PyTypeChecker
            sample = next(
                p
                for p in self.file.infile
                if Path(p).is_file()
            )
        except StopIteration:
            raise FileNotFoundError('No image files found to infer dimension.')
        return iio.imread(sample).shape[1]  # width

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dimension, self.dimension, 3

    @cached_property
    def length(self) -> int:
        """How many input grid comprise a tile of this class"""
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
            .set_axis(self.frame.index)
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
            .set_axis(self.frame.index)
            .sub(self.xtile.min())
        )
        return result

    @File
    def file(self):
        """Namespace for file attributes"""
        # See the following:
        _ = self.file.infile


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
    def from_location(
            cls,
            location,
            zoom: int = None,
    ) -> Self:
        latlon = util.geocode(location)
        result = cls.from_bounds(
            latlon=latlon,
            zoom=zoom
        )
        return result

    @classmethod
    def from_bounds(
            cls,
            latlon: Union[
                str,
                list[float],
            ],
            zoom: int = None,
    ) -> Self:
        """
        Create some base Grid from bounds in lat, lon format and a
        slippy tile zoom level.
        """
        if zoom is None:
            zoom = cfg.zoom
        if isinstance(latlon, str):
            if ', ' in latlon:
                split = ', '
            elif ',' in latlon:
                split = ','
            elif ' ' in latlon:
                split = ' '
            else:
                raise ValueError(
                    "latlon must be a string with coordinates "
                    "separated by either ', ', ',', or ' '."
                )
            latlon = [
                float(x)
                for x in latlon.split(split)
            ]
        gn, gw, gs, ge = latlon
        gn, gs = min(gn, gs), max(gn, gs)
        gw, ge = min(gw, ge), max(gw, ge)
        tw, tn = util.deg2num(gw, gn, zoom=zoom)
        te, ts = util.deg2num(ge, gs, zoom=zoom)
        te, tw = max(te, tw), min(te, tw)
        ts, tn = max(ts, tn), min(ts, tn)
        te += 1
        ts += 1
        tx = np.arange(tw, te)
        ty = np.arange(tn, ts)
        index = pd.MultiIndex.from_product([tx, ty])
        tx = index.get_level_values(0)
        ty = index.get_level_values(1)
        result = cls.from_integers(tx, ty, scale=zoom)
        return result

    @classmethod
    def from_integers(
            cls,
            xtile: Series | Index | np.ndarray,
            ytile: Series | Index | np.ndarray,
            scale: int,
    ) -> Self:
        """
        Construct Grid from integer tile numbers.
        This allows for a non-rectangular grid of grid.
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
        grid = GeoDataFrame(
            data=data,
            index=index,
            geometry=geometry,
            crs=3857
        )
        result = cls(grid)
        result.scale = scale
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
        Construct Grid from ranges of tile numbers.
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

        if not (
            xmin_arr.shape ==
            ymin_arr.shape ==
            xmax_arr.shape ==
            ymax_arr.shape
        ):
            msg = 'xmin, ymin, xmax, ymax must have identical shapes'
            raise ValueError(msg)

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

        mosaic_length = 2 ** abs(self.scale - scale)
        if self.scale < scale:
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

        elif self.scale > scale:
            # into larger tiles
            frame: pd.DataFrame = (
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

        result.__dict__.update(self.__dict__)
        result.scale = scale
        return result


    @classmethod
    def from_rescale(
            cls,
            grid: Grid,
            scale: int,
            fill: bool = True,
    ) -> Self:
        """
        Rescale grid to a new scale.
        If the new scale is larger, the grid are filled with zeros.
        If the new scale is smaller, the grid are downscaled.
        """
        scaled = grid.to_scale(scale, fill=fill)
        xtile = scaled.xtile
        ytile = scaled.ytile
        result = cls.from_integers(
            xtile=xtile,
            ytile=ytile,
            scale=scale,
        )
        assert result.scale == scale
        return result


    def to_padding(self, pad: int = 1) -> Self:
        """ Pad each tile by `pad` tiles in each direction. """
        padded = (
            self
            .to_corners(self.scale)
            .to_padding(pad)
            .to_grid()
            .sort_index()
        )
        assert self.xtile.min() - pad == padded.xtile.min()
        assert self.ytile.min() - pad == padded.ytile.min()
        assert self.xtile.max() + pad == padded.xtile.max()
        assert self.ytile.max() + pad == padded.ytile.max()
        if pad >= 0:
            assert self.index.isin(padded.index).all()
        padded.__dict__.update(self.__dict__)
        return padded

    def to_corners(self, scale: int = None) -> Corners:
        if scale is None:
            scale = self.scale

        length = 2 ** (self.scale - scale)
        result = Corners.from_data(
            xmin=self.xtile.values // length,
            ymin=self.ytile.values // length,
            xmax=(self.xtile.values + 1) // length,
            ymax=(self.ytile.values + 1) // length,
            scale=scale,
            index=self.index,
        )
        return result

    @Cfg
    def cfg(self):
        # This code block is just semantic sugar and does not run.
        # You can access the various configuration options this way:
        _ = self.cfg.zoom
        _ = self.cfg.model.bs_val
        _ = self.cfg.polygon.max_hole_area
        # Please do not set the configuration options directly,
        # you may introduce bugs.

    frame: GeoDataFrame

    @property
    def index(self):
        return self.frame.index

    @property
    def indir(self):
        return self.ingrid.indir

    @property
    def outdir(self):
        return self.ingrid.outdir

    def __len__(self):
        return len(self.frame)

    def pipe(self, *args, **kwargs):
        func = args[0] if args else kwargs.pop('func', None)
        if func is None:
            raise ValueError('func must be provided to pipe')
        result = func(self, *args[1:], **kwargs)
        return result





