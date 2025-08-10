from __future__ import annotations
import geopandas as gpd

import copy
from functools import *
from typing import *

import pandas as pd
from geopandas import GeoDataFrame

from tile2net.grid import static


class FrameWrapper(

):
    frame: pd.DataFrame | gpd.GeoDataFrame

    def __init__(
            self,
            frame: Union[
                gpd.GeoDataFrame,
                pd.DataFrame,
            ] = None,
    ):
        if frame is None:
            frame = gpd.GeoDataFrame()
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

    def __delitem__(self, key):
        del self.frame[key]

    def __setitem__(self, key, value):
        self.frame[key] = value

    def __getitem__(self, item) -> Union[
        pd.Series,
        pd.DataFrame,
        gpd.GeoDataFrame,
        gpd.GeoSeries,
        pd.Index,
        pd.MultiIndex,
    ]:
        return self.frame[item]

    @classmethod
    def from_copy(
            cls,
            frame: pd.DataFrame,
            wrapper: Self,
    ) -> Self:
        result = wrapper.copy()
        result.frame = frame
        return result

    def to_copy(
            self,
            frame=None,
            **kwargs,
    ) -> Self:
        result = self.copy()
        if frame is not None:
            result.frame = frame.copy()
        else:
            result.frame = self.frame.copy()

        result.__dict__.update(**kwargs)
        return result

    def __len__(self):
        return len(self.frame)

    @property
    def crs(self):
        return self.frame.crs

    @property
    def columns(self):
        return self.frame.columns

    @property
    def geometry(self) -> gpd.GeoSeries:
        return self.frame.geometry

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __copy__(self) -> Self:
        result = self.__class__()
        result.__dict__.update(self.__dict__)
        return result

