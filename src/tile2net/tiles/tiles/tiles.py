from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import *

import PIL.Image
import imageio.v3 as iio
import numpy as np
import pandas as pd
import pyproj
import shapely
from PIL import Image

from tile2net.raster import util
from tile2net.tiles import util
from tile2net.tiles.explore import explore
from tile2net.tiles.fixed import GeoDataFrameFixed
from .colormap import ColorMap
from .static import Static
from .tile import Tile

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
            zoom: int,
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
        # This code block is just semantic sugar and does not run.
        # This is a namespace container for static files:
        self.static.hrnet_checkpoint = ...
        self.static.snapshot = ...

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

        mosaic_length = (self.tile.scale - scale) ** 2
        if self.tile.scale < scale:
            # smaller tiles
            smaller = self.index.to_frame()
            larger = smaller // mosaic_length
            topleft = (
                larger
                .drop_duplicates()
                .mul(mosaic_length)
            )
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
            tile_dimension = self.tile.dimension // mosaic_length

        elif self.tile.scale > scale:
            # larger tiles
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
                    .groupby(frame.columns.tolist())
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
            tile_dimension = self.tile.dimension * mosaic_length

        else:
            # same scale
            result = self.copy()
            tile_dimension = self.tile.dimension

        result.attrs.update(self.attrs)
        result.tile.dimension = tile_dimension
        result.tile.scale = scale
        return result
