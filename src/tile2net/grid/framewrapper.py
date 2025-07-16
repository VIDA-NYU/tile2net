from __future__ import annotations

import numpy as np
import pandas as pd

from tile2net.raster import util
from tile2net.tiles import util

from . import util, static
import shapely
import pyproj
from pandas import MultiIndex
from . import frame

import copy
from typing import *

from functools import *

from geopandas import GeoDataFrame
from pandas import Series, Index
import pandas as pd


class FrameWrapper:
    frame: pd.DataFrame

    def __init__(
            self,
            frame: GeoDataFrame,
    ):
        self.frame = frame

    @property
    def index(self):
        return self.frame.index

    def _iloc(self, item) -> Self:
        frame = self.frame.iloc[item]
        result = self.copy()
        result.frame = frame
        return result

    @property
    def iloc(self) -> Self:
        """
        Wrapper for self.frame.iloc.
        self.iloc[...] is equivalent to self.frame.iloc[...] but
        returns a Grid instance instead of a GeoDataFrame.
        """
        return partial(self._iloc)

    def _loc(self, item) -> Self:
        frame = self.frame.loc[item]
        result = self.copy()
        result.frame = frame
        return result

    @property
    def loc(self):
        """
        Wrapper for self.frame.loc.
        self.loc[...] is equivalent to self.frame.loc[...] but
        returns a Grid instance instead of a GeoDataFrame.
        """
        return partial(self._loc)

    def copy(self) -> Self:
        result = copy.copy(self)
        del result.static
        return result

    @static.Static
    def static(self):
        """
        """

    if False:
        def __getitem__(self, item) -> Self:
            ...
