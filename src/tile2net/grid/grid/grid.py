from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import *
from typing import *

import PIL.Image
import imageio.v3 as iio
import math
import numpy as np
import pandas as pd
import pyproj
import shapely
from PIL import Image
from geopandas import GeoDataFrame
from pandas import MultiIndex
from pandas import Series, Index

from tile2net.grid.cfg.logger import logger
from tile2net.grid.explore import explore
from tile2net.raster import util
from .colormap import ColorMap
from .corners import Corners
from .file import File
from .stitcher import Stitcher
from .. import frame
from .. import util
from ..cfg import cfg, Cfg
from ..frame.framewrapper import FrameWrapper

if False:
    import folium
    from ..seggrid.seggrid import SegGrid
    from ..ingrid.ingrid import InGrid
    from ..vecgrid.vecgrid import VecGrid

# thread-local store
tls = threading.local()


class Grid(
    FrameWrapper,
):

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
    def min_scale(self) -> int:
        dim = max(self.r.max(), self.c.max()) + 1
        scale = math.log2(dim)
        scale = math.ceil(scale)
        scale = int(scale)
        result = self.scale - scale

        return result

    @cached_property
    def scale(self) -> int:
        """
        Tile scale; the XYZ scale of the grid.
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
        """How many pixels in a segmentation tile"""
        result = self.ingrid.dimension * self.length
        return result

    # @property
    # def shape(self) -> tuple[int, int, int]:
    #     return self.dimension, self.dimension, self.ingrid.shape[2]

    @property
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension,

    @cached_property
    def length(self) -> int:
        """How many input grid comprise a tile of this class"""
        raise NotImplemented

    @property
    def area(self):
        return self.length ** 2

    @property
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

    @property
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

    @property
    def ingrid(self) -> InGrid:
        return self.instance

    @property
    def seggrid(self) -> SegGrid:
        return self.ingrid.seggrid

    @property
    def vecgrid(self) -> VecGrid:
        return self.ingrid.vecgrid

    @classmethod
    def from_location(
            cls,
            location: str,
            zoom: int = None,
    ) -> Self:
        latlon = util.geocode(location)
        result = cls.from_bounds(
            latlon=latlon,
            zoom=zoom
        )
        result.location = location
        return result

    @cached_property
    def location(self) -> str:
        raise ValueError

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

        elif self.scale > scale:
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

            assert len(self) > len(scaled)
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
        # This code block is just semantic sugar and does not run.
        # You can access the various configuration options this way:
        _ = self.cfg.zoom
        _ = self.cfg.model.bs_val
        _ = self.cfg.polygon.max_hole_area
        # Please do not set the configuration options directly,
        # you may introduce bugs.

    frame: GeoDataFrame

    @property
    def index(self) -> MultiIndex:
        return self.frame.index

    @property
    def indir(self):
        return self.ingrid.indir

    @property
    def outdir(self):
        return self.ingrid.outdir

    @property
    def tempdir(self):
        return self.ingrid.tempdir

    def __len__(self):
        return len(self.frame)

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
    ) -> folium.map:
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
                'to set the inference grid.'
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
            msg = 'You must specify either dimension, length, or mosaic to set the scale.'
            raise ValueError(msg)

        _scale = scale
        scale = max(scale, self.ingrid.min_scale)

        return scale

    def preview(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = 'red',
            show: bool = True,
            files: pd.Series | None = None,
    ) -> PIL.Image.Image:

        # input columns
        if files is None:
            files: pd.Series = self.file.infile
        R: pd.Series = self.r
        C: pd.Series = self.c

        # grid geometry
        dim = self.dimension
        n_rows = int(R.max()) + 1
        n_cols = int(C.max()) + 1
        div_px = 1 if divider else 0

        # full mosaic size before optional down-scaling
        full_w0 = n_cols * dim + div_px * (n_cols - 1)
        full_h0 = n_rows * dim + div_px * (n_rows - 1)

        # scale to maxdim if needed
        scale = 1.0 if max(full_w0, full_h0) <= maxdim else maxdim / max(full_w0, full_h0)

        # derived sizes
        tile_w = max(1, int(round(dim * scale)))
        tile_h = tile_w
        full_w = n_cols * tile_w + div_px * (n_cols - 1)
        full_h = n_rows * tile_h + div_px * (n_rows - 1)

        # canvas
        canvas_col = divider if divider else (0, 0, 0)
        mosaic = Image.new('RGB', (full_w, full_h), color=canvas_col)

        # tile loader
        def load(idx: int) -> tuple[int, int, np.ndarray]:
            arr = iio.imread(files.iat[idx])
            if scale != 1.0:
                arr = np.asarray(
                    Image.fromarray(arr).resize(
                        (tile_w, tile_h),
                        Image.Resampling.LANCZOS,
                    )
                )
            return R.iat[idx], C.iat[idx], arr

        # compose mosaic
        with ThreadPoolExecutor() as pool:
            for r, c, arr in pool.map(load, range(len(files))):
                x0 = c * (tile_w + div_px)
                y0 = r * (tile_h + div_px)
                mosaic.paste(Image.fromarray(arr), (x0, y0))

        # optional popup in PyCharm's SciView (matplotlib)
        if show:
            try:
                import matplotlib.pyplot as plt

                # dpi chosen to avoid oversized windows while preserving sharpness
                dpi = 96
                fig_w_in = max(1.0, full_w / dpi)
                fig_h_in = max(1.0, full_h / dpi)

                plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
                plt.imshow(mosaic)
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.show()

            except Exception:
                # fallback to OS viewer if matplotlib/SciView is unavailable
                import tempfile, os
                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                try:
                    mosaic.save(tmp.name)
                finally:
                    tmp.close()
                Image.open(tmp.name).show()
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        return mosaic

    @classmethod
    def from_empty(cls) -> Self:
        ...

    def _stitch(
            self,
            small_files: pd.Series,
            big_files: pd.Series,
            small_grid: Grid,
            big_grid: Grid,
            r: pd.Series,
            c: pd.Series,
            tile_shape: tuple[int, int],
            mosaic_shape: tuple[int, int],
            background: int = 0,
            force=False,
    ):
        row = r
        col = c
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
                f'{small_grid.__name__}.{small_files.name} '
                f'into {n_total:,} '
                f'{small_grid.__name__}.{big_files.name}'
            )
            logger.info(msg)

        loader = Stitcher(
            infiles=small_files,
            row=row,
            col=col,
            # tile_shape=(*tile_shape, small_grid.shape[2]),
            # mosaic_shape=(*mosaic_shape, small_grid.shape[2]),
            tile_shape=tile_shape,
            mosaic_shape=mosaic_shape,
            # tile_shape=small_grid.shape,
            # mosaic_shape=big_grid.shape,
            outfiles=big_files,
            background=background,
        )

        loader.run(max_workers=os.cpu_count())
        msg = 'Not all stitched mosaics were written to disk.'
        assert big_files.map(os.path.exists).all(), msg

    # grid/grid/grid.py

    def _stitch(
            self,
            small_files: pd.Series,
            big_files: pd.Series,
            small_grid: Grid,
            big_grid: Grid,
            r: pd.Series,
            c: pd.Series,
            background: int = 0,
            force=False,
            **kwargs
    ):

        # keep original row/col
        row = r
        col = c

        # skip mosaics that already exist unless force=True
        if not force:
            loc = ~big_files.map(os.path.exists)
            small_files = small_files.loc[loc]
            row = r.loc[loc]
            col = c.loc[loc]
            big_files = big_files.loc[loc]

        stitched = big_files.drop_duplicates()
        n_missing = len(small_files)
        n_total = len(stitched)

        if n_missing == 0:
            msg = f'All {n_total:,} mosaics are already stitched.'
            logger.info(msg)
        else:
            msg = (
                f'Stitching {n_missing:,} '
                f'{small_grid.__name__}.{small_files.name} '
                f'into {n_total:,} '
                f'{small_grid.__name__}.{big_files.name}'
            )
            logger.info(msg)

            loader = Stitcher(
                infiles=small_files,
                row=row,
                col=col,
                outfiles=big_files,
                background=background,
            )

            loader.run(max_workers=os.cpu_count())
            msg = 'Not all stitched mosaics were written to disk.'
            assert big_files.map(os.path.exists).all(), msg

    @property
    def crs(self):
        return self.crs

    def __delete__(
            self,
            instance: Grid,
    ):
        del instance.frame.__dict__[self.__name__]
        # del instance.__dict__[self.__name__]

    def __set__(
            self,
            instance: Grid,
            value,
    ):
        # instance.__dict__[self.__name__] = value
        instance.frame.__dict__[self.__name__] = value

    @ColorMap
    def colormap(self):
        # This code block is just semantic sugar and does not run.
        # This allows us to apply colormaps to tensors, ndarrays, and images.
        # todo: allow setting custom colormaps
        # See:
        self.colormap.__call__(...)
        self.colormap(...)
