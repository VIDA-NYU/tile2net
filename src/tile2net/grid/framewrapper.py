from __future__ import annotations

import copy
from functools import *
from typing import *

import pandas as pd
from geopandas import GeoDataFrame

from . import static


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

    def __delitem__(self, key):
        del self.frame[key]

    def __setitem__(self, key, value):
        self.frame[key] = value

    @classmethod
    def from_copy(
            cls,
            wrapper: Self,
            frame: pd.DataFrame
    ) -> Self:
        result = wrapper.copy()
        result.frame = frame
        return result

    def to_copy(
            self,
            **kwargs,
    ) -> Self:
        result = self.copy()
        result.__dict__.update(**kwargs)
        return result

