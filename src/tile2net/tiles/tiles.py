from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import *

import PIL.Image
import imageio.v3 as iio
import numpy as np
import pandas as pd
import pyproj
import shapely
from PIL import Image

from tile2net.logger import logger
from tile2net.raster import util
from tile2net.tiles.cfg import cfg
from . import util
from .colormap import ColorMap
from .explore import explore
from .fixed import GeoDataFrameFixed
from .tile import Tile
from .static import Static

if False:
    import folium


class Tiles(
    GeoDataFrameFixed,
):
    gw: pd.Series  # geographic west bound of the tile
    gn: pd.Series  # geographic north bound of the tile
    ge: pd.Series  # geographic east bound of the tile
    gs: pd.Series  # geographic south bound of the tile

    @property
    def xtile(self) -> pd.Index:
        """Tile integer X"""
        return self.index.get_level_values('xtile')

    @property
    def ytile(self) -> pd.Index:
        """Tile integer Y"""
        return self.index.get_level_values('ytile')

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
            tx,
            ty,
            zoom: int = 19,
    ) -> Self:
        """
        Construct Tiles from integer tile numbers.
        This allows for a non-rectangular grid of tiles.
        """
        if isinstance(ty, (pd.Series, pd.Index)):
            tn = ty.values
        elif isinstance(ty, np.ndarray):
            tn = ty
        else:
            raise TypeError('ty must be a Series, Index, or ndarray')
        tn = tn.astype('uint32')

        if isinstance(tx, (pd.Series, pd.Index)):
            tw = tx.values
        elif isinstance(tx, np.ndarray):
            tw = tx
        else:
            raise TypeError('tx must be a Series, Index, or ndarray')
        tw = tw.astype('uint32')

        te = tw + 1
        ts = tn + 1
        names = 'xtile ytile'.split()
        index = pd.MultiIndex.from_arrays([tx, ty], names=names)
        gw, gn = util.num2deg(tw, tn, zoom=zoom)
        ge, gs = util.num2deg(te, ts, zoom=zoom)
        trans = (
            pyproj.proj.Transformer
            .from_crs(4326, 3857, always_xy=True)
            .transform
        )
        pw, pn = trans(gw, gn)
        pe, ps = trans(ge, gs)
        geometry = shapely.box(pw, pn, pe, ps)

        data = dict(
            gw=gw,
            gn=gn,
            ge=ge,
            gs=gs,
        )

        result = cls(
            data=data,
            geometry=geometry,
            index=index,
            # crs=4326,
            crs=3857,
        )
        result.zoom = zoom
        return result

    def with_indir(
            self,
            indir: str,
            name: str = None,
    ) -> Self:
        """
        Assign an input directory to the tiles. The directory must
        implicate the X and Y tile numbers in the file names.

        Good:
            input/dir/x/y/z.png
            input/dir/x/y/.png
            input/dir/x_y_z.png
            input/dir/x_y.png
            input/dir/z/x_y.png
        Bad:
            input/dir/x.png
            input/dir/y

        This will set the input directory and format
        self.with_indir('input/dir/x/y/z.png')

        There is no format specified to, so it will default to
        input/dir/x_y_z.png:
        self.with_indir('input/dir')

        This will fail to explicitly set the format, so it will
        default to input/dir/x/x_y_z.png
        self.with_indir('input/dir/x')
        """
        result = self.copy()
        if name:
            result.name = name
        try:
            result.indir = indir
            indir: Indir = result.indir
            msg = f'Setting input directory to \n\t{indir.original}. '

            logger.info(msg)

        except ValueError as e:
            msg = (
                f'Invalid input directory: {indir}. '
                f'The directory directory must implicate the X and Y '
                f'tile numbers by including `x` and `y` in some format, '
                f'for example: '
                f'input/dir/x/y/z.png or input/dir/x_y_z.png.'
            )
            raise ValueError(msg) from e
        try:
            _ = result.outdir
        except AttributeError:
            msg = (
                f'Output directory not yet set. Based on the input directory, '
                f'setting it to a default value.'
            )
            logger.info(msg)
            result = result.with_outdir()
        return result

    def with_outdir(
            self,
            outdir: Union[str, Path] = None,
    ) -> Self:
        """
        Assign an output directory to the tiles.
        The tiles are saved to the output directory.
        """
        result: Tiles = self.copy()
        if outdir:
            ...
        elif cfg.output_dir:
            outdir = cfg.output_dir
        else:
            if self.name:
                name = self.name
            elif cfg.name:
                name = cfg.name
            elif (
                    self.source
                    and self.source.name
            ):
                name = self.source.name
            else:
                msg = (
                    f'No output directory specified, and unable to infer '
                    f'a name for a temporary output directory. Either '
                    f'set a name, use a source, or specify an output directory.'
                )
                raise ValueError(msg)

            outdir = os.path.join(
                tempfile.gettempdir(),
                'tile2net',
                name,
                'outdir'
            )

        try:
            result.outdir = outdir
        except ValueError:
            retry = os.path.join(outdir, 'z', 'x_y.png')
            result.outdir = retry
            logger.info(f'Setting output directory to \n\t{retry}')
        else:
            logger.info(f'Setting output directory to \n\t{outdir}')

        return result

    @cached_property
    def r(self) -> pd.Series:
        """Row of the tile within the overall grid."""
        ytile = (
            self.ytile
            .to_series()
            .set_axis(self.index)
        )
        result = ytile - ytile.min()
        return result

    @cached_property
    def c(self) -> pd.Series:
        """Column of the tile within the overall grid."""
        xtile = (
            self.xtile
            .to_series()
            .set_axis(self.index)
        )
        result = xtile - xtile.min()
        return result

    @property
    def file(self):
        return self.indir.files()

    @property
    def name(self) -> str:
        return self.cfg.name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise TypeError('Tiles.name must be a string')
        self.cfg.name = value

    def visualize(
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

    def view(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
    ) -> PIL.Image.Image:

        files: pd.Series = self.file
        R: pd.Series = self.r  # 0-based row id
        C: pd.Series = self.c  # 0-based col id

        dim = self.dimension  # original tile side length
        n_rows = int(R.max()) + 1
        n_cols = int(C.max()) + 1
        div_px = 1 if divider else 0

        # full mosaic size before optional down-scaling
        full_w0 = n_cols * dim + div_px * (n_cols - 1)
        full_h0 = n_rows * dim + div_px * (n_rows - 1)

        scale = 1.0
        if max(full_w0, full_h0) > maxdim:
            scale = maxdim / max(full_w0, full_h0)

        tile_w = max(1, int(round(dim * scale)))
        tile_h = tile_w  # square tiles
        full_w = n_cols * tile_w + div_px * (n_cols - 1)
        full_h = n_rows * tile_h + div_px * (n_rows - 1)

        canvas_col = divider if divider else (0, 0, 0)
        mosaic = Image.new('RGB', (full_w, full_h), color=canvas_col)

        def load(idx: int) -> tuple[int, int, np.ndarray]:
            arr = iio.imread(files.iat[idx])
            if scale != 1.0:
                arr = np.asarray(
                    Image.fromarray(arr).resize(
                        (tile_w, tile_h), Image.Resampling.LANCZOS
                    )
                )
            return R.iat[idx], C.iat[idx], arr

        with ThreadPoolExecutor() as pool:
            for r, c, arr in pool.map(load, range(len(files))):
                x0 = c * (tile_w + div_px)
                y0 = r * (tile_h + div_px)
                mosaic.paste(Image.fromarray(arr), (x0, y0))

        return mosaic

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
        instance.attrs[self.__name__] = value

    @Static
    def static(self):
        ...
