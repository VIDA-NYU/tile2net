from __future__ import annotations
import copy
import functools

import math

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
from . import tile

if False:
    import folium
    from .tiles import Tiles
    from ..intiles import InTiles
    from ..segtiles import SegTiles
    from ..vectiles import VecTiles


#
# import pandas as pd


def __get__(
        self: Tile,
        instance: Corners,
        owner,
) -> Tile:
    self.corners = instance
    return copy.copy(self)


class Tile(

):
    locals().update(
        __get__=__get__
    )
    corners: Corners = None

    @property
    def tiles(self):
        return self.corners

    def __init__(self, *args):
        ...

    @tile.cached_property
    def area(self):
        return self.width * self.height

    @tile.cached_property
    def width(self) -> int:
        """How many input tiles comprise a tile of this class"""
        return self.corners.xmax - self.corners.xmin

    @tile.cached_property
    def height(self) -> int:
        """Tile height in pixels"""
        return self.corners.ymax - self.corners.ymin

    @tile.cached_property
    def scale(self) -> int:
        """Tile scale"""

    def __set_name__(self, owner, name):
        self.__name__ = name


class Corners(
    GeoDataFrameFixed
):
    xmin: pd.Series
    ymin: pd.Series
    xmax: pd.Series
    ymax: pd.Series

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
        ...

    @classmethod
    def from_data(
            cls,
            xmin: Union[np.ndarray, pd.Series],
            ymin: Union[np.ndarray, pd.Series],
            xmax: Union[np.ndarray, pd.Series],
            ymax: Union[np.ndarray, pd.Series],
            scale: int,
            index: pd.MultiIndex,
    ) -> Self:
        data = dict(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        result = cls(data, index=index)
        result.tile.scale = scale
        return result

    def to_scale(self, scale: int) -> Self:
        """Returns a new Corners object with corners rescaled"""
        length = 2 ** (self.tile.scale - scale)
        result = self.from_data(
            xmin=self.xmin // length,
            ymin=self.ymin // length,
            xmax=self.xmax // length,
            ymax=self.ymax // length,
            scale=scale,
            index=self.index,
        )
        return result

    def to_padding(self, pad: int = 1) -> Self:
        """Returns a new Corners object with padding added to each corner"""
        # todo: handle negative padding
        assert pad >= 0
        result = self.from_data(
            xmin=self.xmin - pad,
            ymin=self.ymin - pad,
            xmax=self.xmax + pad,
            ymax=self.ymax + pad,
            scale=self.tile.scale,
            index=self.index,
        )
        return result

    def to_tiles(self, drop_duplicates=True) -> Tiles:
        """Creates a range of tiles for each set of corner extrema"""
        from .tiles import Tiles
        result = Tiles.from_ranges(
            xmin=self.xmin,
            ymin=self.ymin,
            xmax=self.xmax,
            ymax=self.ymax,
            scale=self.tile.scale,
        )
        if drop_duplicates:
            loc = ~result.index.duplicated()
            result = result.loc[loc]
        return result
