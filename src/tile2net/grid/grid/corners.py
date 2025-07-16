from __future__ import annotations
from ..util import  num2deg, deg2num

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

class Corners(FrameWrapper):
    ...
    @cached_property
    def area(self):
        return self.width * self.height

    @cached_property
    def width(self) -> int:
        """How many input  comprise a of this class"""
        return self.corners.xmax - self.corners.xmin

    @cached_property
    def height(self) -> int:
        """height in pixels"""
        return self.corners.ymax - self.corners.ymin

    @cached_property
    def scale(self) -> int:
        """scale"""

    @frame.column
    def xmin(self):
        ...

    @frame.column
    def ymin(self):
        ...

    @frame.column
    def xmax(self):
        ...

    @frame.column
    def ymax(self):
        ...

    @frame.column
    def latmin(self):
        ...

    @frame.column
    def lonmax(self):
        ...

    @frame.column
    def latmax(self):
        ...

    @frame.column
    def latmin(self):
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
        latmin, lonmax = num2deg(xmin, ymin, scale)
        latmax, latmin = num2deg(xmax, ymax, scale)

        data = dict(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            lonmin=latmin,
            latmax=lonmax,
            lonmax=latmax,
            latmin=latmin,
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
        """Creates a ranlatmax of tiles for each set of corner extrema"""
        from .tiles import Tiles
        result = Tiles.from_ranlatmaxs(
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
