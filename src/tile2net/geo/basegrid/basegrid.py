from __future__ import annotations

import hashlib
import math
import os
import tempfile
from functools import *
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import shapely
import tqdm
from PIL import ImageColor, Image
from geopandas import GeoDataFrame
from pandas import MultiIndex, Series, Index

from tile2net.grid.basegrid import basegrid
from tile2net.grid.basegrid.file import File
from tile2net.grid.explore import explore
from tile2net.grid import frame, util
from tile2net.grid.cfg import cfg, Cfg
from tile2net.grid.cfg.logger import logger
from tile2net.grid.loaders.dataloader import BaseDataLoader
from tile2net.grid.loaders.datawrapper import DataWrapper
from tile2net.grid.loaders.rescale import RescaleDataSet
from tile2net.grid.loaders.stitch import StitchWriterDataSet
from tile2net.grid.loaders.unstitch import UnstitchDataSet, UnstitchDataWrapper
from tile2net.grid.sampler.benchmark import Benchmark
from tile2net.geo.basegrid.corners import Corners

if TYPE_CHECKING:
    import folium
    from tile2net.geo.seggrid.seggrid import SegGrid
    from tile2net.geo.grid import Grid
    from tile2net.geo.vecgrid.vecgrid import VecGrid
    from tile2net.grid.dir.outdir import Outdir


class BaseGrid(
    basegrid.BaseGrid
):

    @frame.column
    def lonmax(self):
        """
        Maximum longitude of the tile.
        
        Example:
            >>> grid: Grid
            >>> grid.lonmax
            xtile   ytile
            317280  387840   -7.911500e+06
        """

    @frame.column
    def lonmin(self):
        """
        Minimum longitude of the tile.
        
        Example:
            >>> grid: Grid
            >>> grid.lonmin
            xtile   ytile
            317280  387840   -7.911538e+06
        """

    @frame.column
    def latmax(self):
        """
        Maximum latitude of the tile.
        
        Example:
            >>> grid: Grid
            >>> grid.latmax
            xtile   ytile
            317280  387840    5.214840e+06
        """

    @frame.column
    def latmin(self):
        """
        Minimum latitude of the tile.
        
        Example:
            >>> grid: Grid
            >>> grid.latmin
            xtile   ytile
            317280  387840    5.214802e+06
        """

    @frame.index
    def xtile(self):
        """
        X tile coordinate of the tile.
        
        Example:
            >>> grid: Grid
            >>> grid.xtile
            Index([317280, 317280, 317280, 317280, 317280, 317280, 317280, 317280, 317280,
        """

    @frame.index
    def ytile(self) -> Index | Series:
        """
        Y tile coordinate of the tile.
        
        Example:
            >>> grid: Grid
            >>> grid.ytile
            Index([387840, 387841, 387842, 387843, 387844, 387845, 387846, 387847, 387848,
        """


    @cached_property
    def hash(self) -> str:
        """Hash of the Tiles in the grid and the configuration."""
        pairs = (
            self.index
            .to_frame(index=False)  # -> DataFrame with ['xtile', 'ytile']
            .astype({'xtile': 'int64', 'ytile': 'int64'}, copy=False)
            .to_numpy(copy=False)  # -> (n, 2) int64 ndarray
        )
        tiles = hashlib.blake2b(
            np.ascontiguousarray(pairs).tobytes(),
            digest_size=8,
        ).hexdigest()
        cfg = self.cfg.hash()
        result = f'{tiles}-{cfg}'
        return result

    @property
    def min_scale(self) -> int:
        dim = max(self.r.max(), self.c.max()) + 1
        scale = math.log2(dim)
        scale = math.ceil(scale)
        scale = int(scale)
        result = self.scale - scale

        return result

    scale: int
    """Tile scale; the XYZ scale of the grid.
    Higher value means smaller area.
    """

    @cached_property
    def zoom(self) -> int:
        """
        Zoom level of the tile.
        """
        return self.scale

    @property
    def dimension(self):
        """
        Pixel dimension of each tile

        Computed as grid.dimension * self.length. For example, if Grid tiles
        are 256x256 pixels and length is 4, tiles are 1024x1024 pixels.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.dimension
            1024
        """
        result = self.grid.dimension * self.length
        return result

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of a tile in pixels"""
        return self.dimension, self.dimension

    @cached_property
    def length(self) -> int:
        """
        Number of Grid tiles that comprise one dimension of this tile

        Computed as 2^(grid.scale - self.scale). For example, if Grid uses zoom 20
        and this grid uses zoom 18, each tile is 2^2 = 4 Grid tiles wide.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.length
            4
        """
        raise ValueError('Not yet set')

    @property
    def area(self) -> int:
        """Number of in-tiles that comprise one tile in this grid."""
        return self.length ** 2

    @property
    def r(self) -> Series:
        """ Row of the tile within the overall grid. """
        result = (
            self.ytile
            .to_series()
            .set_axis(self.frame.index)
            .sub(self.ytile.min())
        )
        return result

    @property
    def c(self) -> Series:
        """Column of the tile within the overall grid"""
        result = (
            self.xtile
            .to_series()
            .set_axis(self.frame.index)
            .sub(self.xtile.min())
        )
        return result

    @File
    def file(self):
        """
        Namespace for file attributes

        Example:
            >>> self.file.Static
            xtile   ytile
            317280  387840    /home/<user>/tile2net/ma/grid/static/20/31...
                    387841    /home/<user>/tile2net/ma/grid/static/20/31...
        """

    @property
    def grid(self) -> Grid:
        """Reference to the Grid instance"""
        return self.instance

    @property
    def seggrid(self) -> SegGrid:
        """Reference to the SegGrid instance"""
        return self.grid.seggrid

    @property
    def vecgrid(self) -> VecGrid:
        """Reference to the VecGrid instance"""
        return self.grid.vecgrid

    location: str = None
    """Location passed by the user when instantiating the Grid"""

    @classmethod
    def from_bounds(
            cls,
            bounds: Union[
                str,
                list[float],
                list[int],
                tuple[float, float, float, float],
                tuple[int, int, int, int],
            ],
            zoom: int = None,
    ) -> Self:
        """
        Create some base Grid from bounds in lat, lon format and a
        slippy tile zoom level.
        """
        if zoom is None:
            zoom = cfg.zoom

        if isinstance(bounds, str):
            if ', ' in bounds:
                split = ', '
            elif ',' in bounds:
                split = ','
            elif ' ' in bounds:
                split = ' '
            else:
                raise ValueError(
                    "latlon must be a string with coordinates "
                    "separated by either ', ', ',', or ' '."
                )
            bounds = [
                float(x)
                for x in bounds.split(split)
            ]

        if isinstance(bounds, (tuple, list)):
            if len(bounds) != 4:
                raise ValueError('latlon must have exactly 4 elements')

            is_all_int = all(
                isinstance(x, (int, np.integer)) and not isinstance(x, bool)
                for x in bounds
            )
            is_all_float = all(
                isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
                for x in bounds
            )

            if not is_all_float:
                types = {type(x) for x in bounds}
                raise TypeError(
                    'latlon elements must be numeric (int or float), '
                    f'got types: {types}'
                )

            if is_all_int:
                if zoom is None:
                    raise ValueError('zoom must be specified when using xtile, ytile bounds')
                xmin, ymin, xmax, ymax = bounds
                tx = np.arange(xmin, xmax)
                ty = np.arange(ymin, ymax)
                index = pd.MultiIndex.from_product([tx, ty])
                tx = index.get_level_values(0)
                ty = index.get_level_values(1)
                result = cls.from_integers(tx, ty, scale=zoom)
                return result

            bounds = list(bounds)

        gn, gw, gs, ge = bounds
        gn, gs = min(gn, gs), max(gn, gs)
        gw, ge = min(gw, ge), max(gw, ge)
        tw, tn = util.lonlat2xy(gw, gn, zoom=zoom)
        te, ts = util.lonlat2xy(ge, gs, zoom=zoom)
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
        tn = tn.astype(np.uint32)

        if isinstance(xtile, (Series, Index)):
            tw = xtile.values
        elif isinstance(xtile, np.ndarray):
            tw = xtile
        else:
            raise TypeError('xtile must be a Series, Index, or ndarray')
        tw = tw.astype(np.uint32)

        te = tw + 1
        ts = tn + 1
        names = 'xtile ytile'.split()
        index = MultiIndex.from_arrays([xtile, ytile], names=names)
        gw, gn = util.xy2lonlat(tw, tn, zoom=scale)
        ge, gs = util.xy2lonlat(te, ts, zoom=scale)
        trans = (
            pyproj.proj.Transformer
            .from_crs(4326, 3857, always_xy=True)
            .transform
        )
        pw, pn = trans(gw, gn)
        pe, ps = trans(ge, gs)
        geometry = shapely.box(pw, pn, pe, ps)

        data = dict(
            lonmin=pw,
            latmax=pn,
            lonmax=pe,
            latmin=ps,
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

        xgrid: list[np.ndarray] = []
        ygrid: list[np.ndarray] = []

        for x0, y0, x1, y1 in zip(xmin_arr, ymin_arr, xmax_arr, ymax_arr):
            xs = np.arange(x0, x1, dtype='uint32')
            ys = np.arange(y0, y1, dtype='uint32')
            gx, gy = np.meshgrid(xs, ys)  # shape (len(ys), len(xs))
            xgrid.append(gx.ravel())
            ygrid.append(gy.ravel())

        xtile_all = np.concatenate(xgrid)
        ytile_all = np.concatenate(ygrid)

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
        Rescale the Grid to a new slippy-tile scale

        scale:
            new scale of grid
        fill:
            if True, fills missing grid when making larger grid.
            else, the larger grid that have missing grid are dropped.
        """

        mosaic_length = 2 ** abs(self.scale - scale)
        if self.scale < scale:
            # into smaller grid
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
            scaled = self.from_integers(
                xtile,
                ytile,
                scale
            )
            assert len(scaled) == len(self) * (mosaic_length ** 2)
            assert not scaled.index.duplicated().any()

            result = self.copy()
            result.__dict__.update(scaled.__dict__)

        elif self.scale >= scale:
            # into larger grid
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

            scaled = self.from_integers(
                frame.xtile,
                frame.ytile,
                scale
            )

            assert len(self) >= len(scaled)
            assert not scaled.index.duplicated().any()

            result = self.copy()
            result.__dict__.update(scaled.__dict__)

        else:
            # same scale
            result = self.copy()

        return result

    @classmethod
    def from_rescale(
            cls,
            grid: BaseGrid,
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
        """ Pad each tile by `pad` grid in each direction. """
        filled = (
            self
            .to_corners(self.scale)
            .to_padding(pad)
            .to_grid()
            .pipe(self.from_wrapper)
        )
        assert isinstance(filled, self.__class__)
        filled.frame = filled.frame.sort_index()

        assert self.xtile.min() - pad == filled.xtile.min()
        assert self.ytile.min() - pad == filled.ytile.min()
        assert self.xtile.max() + pad == filled.xtile.max()
        assert self.ytile.max() + pad == filled.ytile.max()
        if pad >= 0:
            assert self.index.isin(filled.index).all()

        result = self.copy()
        result.__dict__.update(filled.__dict__)

        return filled

    def to_corners(self, scale: int = None) -> Corners:
        if scale is None:
            scale = self.scale

        length = 2 ** (self.scale - scale)
        xmin = (
            self.xtile.values
            .__floordiv__(length)
            .astype('uint32')
        )
        ymin = (
            self.ytile.values
            .__floordiv__(length)
            .astype('uint32')
        )
        xmax = (
            (self.xtile.values + 1)
            .__floordiv__(length)
            .astype('uint32')
        )
        ymax = (
            (self.ytile.values + 1)
            .__floordiv__(length)
            .astype('uint32')
        )
        result = Corners.from_data(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            scale=scale,
            index=self.index,
        )
        return result

    @Cfg
    def cfg(self):
        """
        Namespace container for configuration options of a Grid.

        Example:
            >>> grid: BaseGrid
            # access zoom level config
            >>> grid.cfg.zoom
            20
            # access batch size config
            >>> grid.cfg.validation.batch_size
            32
            # access polygon param
            >>> grid.cfg.polygon.max_hole_area
            1000
        """

    @property
    def index(self) -> MultiIndex:
        return self.frame.index

    @property
    def outdir(self) -> Outdir:
        """
        Output in which the results, such as annotated images and geometry, will be stored:
        Example:
            >>> grid: Grid
            >>> grid.outdir
            Outdir(
                format='/home/<user>/tile2net/{z}/{x}_{y}',
                dir='/home/<user>/tile2net',
                original='/home/<user>/tile2net/z/x_y',
                suffix='z/x_y'
            )

        Setting the output directory:
        >>> grid: Grid
        >>> grid = grid.set_outdir('/path/to/output')
        """

        return self.grid.outdir

    @property
    def tempdir(self):
        """
        Temporary directory for intermediate processing files.

        Example:
            >>> grid: Grid
            >>> grid.tempdir
            Tempdir(
                dir='/tmp/tile2net/ma/grid/static'
                original='/tmp/tile2net/ma/grid/static/z/x_y',
            )
        """
        return self.grid.tempdir

    def __len__(self):
        return len(self.frame)

    def __repr__(self):
        result = f'{self.__class__.__qualname__}:\n\n'
        try:
            if self.location:
                result += (
                    f'Location: \n\t'
                    f'{self.location}\n'
                )
        except Exception:
            ...
        try:
            ymin, xmin, ymax, xmax = self.lat_lon
            result += (
                f'Lat/Lon Bounds: \n\t'
                f'({ymin:.4f}, {xmin:.4f}, {ymax:.4f}, {xmax:.4f})\n'
            )
        except Exception:
            ...
        try:
            xmin, ymin, xmax, ymax = self.xtile_ytile
            result += (
                f'XTile/YTile Bounds: \n\t'
                f'({xmin}, {ymin}, {xmax}, {ymax})\n'
            )
        except Exception:
            ...
        try:
            result += (
                f'Scale: \n\t'
                f'{self.scale}\n'
            )
        except Exception:
            ...
        try:
            result += (
                f'Source: \n\t'
                f'{self.grid.source}\n'
            )
        except Exception:
            ...
        result += f'\n'
        result += self.frame.__repr__()
        return result

    def pipe(self, *args, **kwargs):
        func = args[0] if args else kwargs.pop('func', None)
        if func is None:
            raise ValueError('func must be provided to pipe')
        result = func(self, *args[1:], **kwargs)
        return result

    def explore(
            self,
            *args,
            loc=None,
            tile_color='grey',
            subset_color='yellow',
            tiles: str = 'cartodbdark_matter',
            m=None,
            dissolve: bool = False,
            **kwargs
    ) -> folium.Map:
        import folium
        frame = self.frame
        if dissolve:
            frame = frame.dissolve()
        if loc is None:
            m = explore(
                # self,
                frame,
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
            grid = frame.loc[loc]
            loc = ~self.index.isin(grid.index)
            m = explore(
                frame.loc[loc],
                color=tile_color,
                name='grid',
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
                grid,
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

    def _to_scale(
            self,
            dimension: int = None,
            length: int = None,
            scale: int = None,
            mosaic: int = None,
    ) -> int:
        """Unifies varying methods of defining tile scales to the slippy-zoom-based scale integer. """

        n = sum(
            arg is not None
            for arg in (dimension, mosaic, scale, length)
        )
        if n != 1:
            msg = (
                'You must specify exactly one of dimension, length, mosaic, or scale '
                'to set the inference grid.'
            )
            raise ValueError(msg)

        if scale:
            ...

        # get scale from dimension, length, or mosaic
        elif dimension:

            if (
                    not isinstance(dimension, int)
                    or dimension <= 0
                    or (dimension & (dimension - 1)) != 0
            ):
                raise ValueError('Dimension must be a positive power of 2.')
            dscale = int(math.log2(self.dimension / dimension))
            scale = self.scale + dscale

        elif length:
            if (
                    not isinstance(length, int)
                    or length <= 0
                    or (length & (length - 1)) != 0
            ):
                raise ValueError('Length must be a positive power of 2.')
            scale = self.scale - int(math.log2(length))

        elif mosaic:
            if (
                    not isinstance(mosaic, int)
                    or mosaic <= 0
                    or (mosaic & (mosaic - 1)) != 0
            ):
                raise ValueError('Mosaic must be a positive power of 2.')
            marea = int(math.log2(mosaic))
            dscale = int(math.sqrt(marea))
            scale = self.scale - dscale

        else:
            msg = 'You must specify either dimension, length, scale, or mosaic to set the scale.'
            raise ValueError(msg)

        _scale = scale
        scale = max(scale, self.grid.min_scale)

        return scale

    def _stitch2file(
            self,
            mosaics: Union[pd.Series, Any],
            tiles: Union[pd.Series, Any],
            row: Union[pd.Series, Any],
            col: Union[pd.Series, Any],
            background: int = 0,
            force=False,
    ):
        """
        Stitch multiple input tiles into larger mosaic images and write them to disk.

        Takes individual tile file paths and combines them into larger mosaics based on
        their row/column positions. Skips existing mosaics unless force=True.

        Args:
            tiles:
                Series of input tile file paths to be stitched together.
            mosaics:
                Series of output mosaic file paths where stitched images will be saved.
            row:
                Row indices indicating vertical position of each tile within its mosaic.
            col:
                Column indices indicating horizontal position of each tile within its mosaic.
            background:
                Pixel value to use for empty/background areas in mosaics.
            force:
                If True, regenerate all mosaics even if they already exist on disk.
            **kwargs:
                Additional arguments passed to the stitching process.

        See also:
            Stitching and serializing static imagery from the source scale to the segmentation scale:
            >>> VecGrid.file.static
        """

        # skip mosaics that already exist unless force=True
        if not force:
            loc = ~mosaics.map(os.path.exists)
            tiles = tiles.loc[loc]
            row = row.loc[loc]
            col = col.loc[loc]
            mosaics = mosaics.loc[loc]

        stitched = mosaics.drop_duplicates()
        n_missing = len(tiles)
        n_total = len(stitched)

        if n_missing == 0:
            msg = f'All {n_total:,} mosaics are already stitched.'
            logger.info(msg)
        else:
            msg = (
                f'Stitching {n_missing:,} '
                f'{self.__class__.__name__}.{tiles.name} '
                f'into {n_total:,} '
                f'{self.__class__.__name__}.{mosaics.name}'
            )
            logger.info(msg)

            wrapper = DataWrapper.from_columns(
                image_path=tiles,
                index=mosaics,
                row=row,
                col=col,
                background=background,
            )
            dataset = StitchWriterDataSet(wrapper)
            batch_size = max(1, os.cpu_count())
            loader = BaseDataLoader(dataset, batch_size)

            total = loader.wrapper.index.nunique()

            pbar = tqdm.tqdm(
                total=total,
                # desc=f'vecgrid.{self.vectorize.__name__}()',
                desc=f'Stitching to file',
                unit=f' {mosaics.name}',
                smoothing=0.01,
                mininterval=10,
            )

            with self.grid.cfg, pbar, self.sampler:
                for minibatch in loader:
                    pbar.update(len(minibatch))

            msg = 'Not all stitched mosaics were written to disk.'
            assert mosaics.map(os.path.exists).all(), msg

    def _unstitch2file(
            self,
            mosaics: Union[pd.Series, Any],
            tiles: Union[pd.Series, Any],
            row: Union[pd.Series, Any],
            col: Union[pd.Series, Any],
            force: bool = False,
    ):
        """
        Split mosaic images into individual tiles and write them to disk.

        Performs the inverse of _stitch2file: reads mosaic images and extracts
        tiles at specified row/column positions, writing each tile to its
        corresponding output path.

        Args:
            mosaics:
                Series of input mosaic file paths to be split into tiles.
            tiles:
                Series of output tile file paths where extracted tiles will be saved.
            row:
                Row indices indicating vertical position of each tile within its mosaic.
            col:
                Column indices indicating horizontal position of each tile within its mosaic.
            force:
                If True, regenerate all tiles even if they already exist on disk.
            **kwargs:
                Additional arguments (unused, for API consistency).

        See also:
            Unstitching predictions from the segmentation scale to the source scale:
            >>> Grid.file.pred
        """

        # skip tiles that already exist unless force=True
        if not force:
            loc = ~tiles.map(os.path.exists)
            mosaics = mosaics.loc[loc]
            tiles = tiles.loc[loc]
            row = row.loc[loc]
            col = col.loc[loc]

        unique_mosaics = mosaics.drop_duplicates()
        n_missing = len(tiles)
        n_mosaics = len(unique_mosaics)

        if n_missing == 0:
            msg = f'All {len(tiles):,} tiles already exist.'
            logger.info(msg)
        else:
            msg = (
                f'Unstitching {n_mosaics:,} '
                f'{self.__class__.__name__}.{mosaics.name} '
                f'into {n_missing:,} '
                f'{self.__class__.__name__}.{tiles.name}'
            )
            logger.info(msg)

            wrapper = UnstitchDataWrapper.from_columns(
                image_path=mosaics,
                outfile=tiles,
                index=mosaics,
                row=row,
                col=col,
            )
            dataset = UnstitchDataSet(
                wrapper=wrapper,
                write=UnstitchDataSet.write_image,
            )
            loader = dataset.loader()

            bar = tqdm.tqdm(
                total=n_mosaics,
                desc=f'Unstitching to file',
                unit=f' {mosaics.name}',
                smoothing=0.01,
                mininterval=10,
            )

            with self.grid.cfg, bar, self.sampler:
                for minibatch in loader:
                    bar.update(len(minibatch))

            msg = 'Not all tiles were written to disk.'
            assert tiles.map(os.path.exists).all(), msg

    @property
    def crs(self):
        return self.crs

    def __delete__(
            self,
            instance: BaseGrid,
    ):
        try:
            del instance.frame.__dict__[self.__name__]
        except KeyError:
            ...

    def __set__(
            self,
            instance: BaseGrid,
            value,
    ):
        # todo: maybe should be copy
        instance.frame.__dict__[self.__name__] = value

    @cached_property
    def colormap(self):
        """
        Callable which applies colormaps to tensors, ndarrays, and images.

        Update cfg.label2color to change the colormap:
            >>> self.cfg.label2color
            {'sidewalk': 'red',
             'road': 'cyan',
             'crosswalk': 'yellow',
             'curb': 'blue',
             'void': 'black'}

        Example:
            >>> self.colormap
            ColorMap(
              0 -> [255, 0, 0]   (sidewalk -> red)
              1 -> [0, 255, 255] (road -> cyan)
              2 -> [255, 255, 0] (crosswalk -> yellow)
              3 -> [0, 0, 0]     (void -> black)
            )

            >>> self.colormap(np.array([[0,1,2]]))
            Out[11]:
            array([[[  0,   0, 255],
                    [  0, 128,   0],
                    [255,   0,   0]]], dtype=uint8)
        """
        return self.cfg.colormap

    @cached_property
    def time_usage(self) -> float:
        return 0

    @cached_property
    def disk_usage(self) -> int:
        return 0

    @cached_property
    def sampler(self) -> Benchmark:
        result = Benchmark(include_gpu=True)
        return result

    @cached_property
    def lat_lon(self) -> tuple[float, float, float, float]:
        """
        Overall bounding box of the grid in lat/lon format.

        Returns:
            Tuple of (lat_north, lon_west, lat_south, lon_east) covering all tiles.

        Example:
            >>> grid.lat_lon
            (42.3601, -71.0589, 42.3551, -71.0539)
        """
        xmin = self.xtile.min()
        ymin = self.ytile.min()
        xmax = self.xtile.max() + 1
        ymax = self.ytile.max() + 1

        lon_west, lat_north = util.xy2lonlat(xmin, ymin, zoom=self.scale)
        lon_east, lat_south = util.xy2lonlat(xmax, ymax, zoom=self.scale)

        return float(lat_north), float(lon_west), float(lat_south), float(lon_east)

    @cached_property
    def xtile_ytile(self) -> tuple[int, int, int, int]:
        """
        Overall bounding box of the grid in xtile/ytile format.

        Returns:
            Tuple of (xtile_min, ytile_min, xtile_max, ytile_max) covering all tiles.

        Example:
            >>> grid.xtile_ytile
            (317280, 387840, 317281, 387841)
        """
        xmin = int(self.xtile.min())
        ymin = int(self.ytile.min())
        xmax = int(self.xtile.max()) + 1
        ymax = int(self.ytile.max()) + 1

        return xmin, ymin, xmax, ymax

    def preview(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = 'red',
            show: bool = True,
            files: Optional[pd.Series] = None,
    ) -> Image.Image:
        # todo: perhaps make this a method for a column subclass, so we have:
        #   grid.file.static.preview()
        #   seggrid.file.pred.preview()
        """
        Generate a mosaic preview of all tiles in the grid.

        Args:
            maxdim: Maximum dimension for the output image
            divider: Color name for tile divider lines
            show: Display the preview automatically
            files: Optional custom file paths to preview

        Returns:
            PIL Image containing the tile mosaic
        """
        # todo: divider isn't showing up

        # grid geometry
        dim = self.dimension
        R: pd.Series = self.r
        C: pd.Series = self.c

        # total rows/cols
        n_rows = int(R.max()) + 1
        n_cols = int(C.max()) + 1

        # compute full-res canvas size assuming 1px divider lines between tiles
        # (the dataset you're using below is already aware of 'divider' as background)
        w = n_cols * dim + n_cols
        h = n_rows * dim + n_rows
        m = max(w, h)

        # downscale factor to respect maxdim
        if m <= maxdim:
            scale = 1.0
        else:
            scale = maxdim / m

        wrapper = DataWrapper.from_columns(
            image_path=self.file.static,
            index=self.index,
            row=self.r,
            col=self.c,
            background=divider,
            force=True,
        )

        # dataset/loader provide batches with x0, y0, arr already scaled
        dataset = RescaleDataSet(wrapper, scale=scale, dim=dim)
        loader = dataset.loader

        # base color for the mosaic background
        base_rgb = ImageColor.getrgb(divider or 'black')

        # allocate mosaic as RGB uint8 and fill with base color
        # shape uses scaled dimensions derived implicitly from dataset outputs;
        # we allocate using the scaled canvas size to avoid incremental growth.
        # Scale the nominal w,h computed above.

        h = max(dataset.y1) + 1
        w = max(dataset.x1) + 1
        mosaic_np = np.empty((h, w, 3), dtype=np.uint8)
        mosaic_np[...] = base_rgb

        # paste tiles using numpy copy semantics
        for batch in loader:
            x0 = batch.x0
            y0 = batch.y0
            arr = batch.arr
            x1 = batch.x1
            y1 = batch.y1

            # ensure uint8 RGB
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)

            if arr.ndim == 2:
                # grayscale → RGB
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                # drop alpha
                arr = arr[:, :, :3]
            elif arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f'Unexpected tile shape: {arr.shape!r}')
            h_tile = arr.shape[0]
            w_tile = arr.shape[1]

            # bounds check (defensive; no-op if in-bounds)
            if (
                    y0 < 0
                    or x0 < 0
                    or y0 + h_tile > mosaic_np.shape[0]
                    or x0 + w_tile > mosaic_np.shape[1]
            ):
                raise ValueError(
                    f'Tile at ({x0},{y0}) with size ({w_tile},{h_tile}) '
                    f'exceeds mosaic bounds ({mosaic_np.shape[1]},{mosaic_np.shape[0]}).'
                )

            # fast in-place write
            np.copyto(mosaic_np[y0:y1, x0:x1, :], arr)

        # convert to PIL for downstream consumers
        mosaic_im = Image.fromarray(mosaic_np, mode='RGB')

        # optional display (PyCharm SciView / matplotlib)
        if show:
            try:
                # keep DPI moderate to avoid giant windows
                dpi = 96
                fig_w_in = max(1.0, mosaic_im.width / dpi)
                fig_h_in = max(1.0, mosaic_im.height / dpi)

                plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
                plt.imshow(mosaic_im)
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.show()

            except Exception:
                # fallback to OS image viewer
                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                try:
                    mosaic_im.save(tmp.name)
                finally:
                    tmp.close()

                try:
                    with Image.open(tmp.name) as im:
                        im.show()
                finally:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass

        return mosaic_im
