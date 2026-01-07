from __future__ import annotations

import copy
import os
from functools import *
from typing import *

import geopandas as gpd
import pandas as pd

from tile2net.grid.frame.namespace import namespace
from .wrapper import Wrapper


class Loc:
    # @weak.property
    # def instance(self) -> FrameWrapper:
    #     ...

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    def __get__(
            self,
            instance,
            owner
    ):
        self.instance = instance
        return copy.copy(self)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __getitem__(self, item):
        result = (
            getattr(self.instance.frame, self.__name__)
            [item]
            .pipe(self.instance.from_frame, self.instance)
        )
        return result

    def __setitem__(self, key, value):
        getattr(self.instance.frame, self.__name__)[key] = value


class FrameWrapper(
    Wrapper,
    namespace,
):
    frame: Union[
        gpd.GeoDataFrame,
        pd.DataFrame,
    ]

    def __init__(
            self,
            frame: Union[
                gpd.GeoDataFrame,
                pd.DataFrame,
            ] = None,
            *args,
            **kwargs
    ):
        super().__init__(frame, *args, **kwargs)
        if frame is None:
            self.frame = gpd.GeoDataFrame()
        elif isinstance(frame, (pd.DataFrame, gpd.GeoDataFrame)):
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
        return result

    def __delitem__(self, key):
        del self.frame[key]

    def __setitem__(self, key, value):
        self.frame[key] = value

    @classmethod
    def from_frame(
            cls,
            frame: pd.DataFrame,
            wrapper: Self = None,
    ) -> Self:
        result = cls()
        if wrapper is not None:
            result.__dict__.update(wrapper.__dict__)
        result.frame = frame.copy()
        return result

    @classmethod
    def from_wrapper(
            cls,
            wrapper: Self,
            frame: pd.DataFrame = None,
    ) -> Self:
        result = cls()
        result.__dict__.update(wrapper.__dict__)
        if frame is not None:
            result.frame = frame.copy()
        else:
            result.frame = result.frame.copy()
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

    def __copy__(self) -> Self:
        result = self.__class__()
        result.__dict__.update(self.__dict__)
        return result

    @Loc
    def loc(self):
        ...

    @Loc
    def iloc(self):
        ...

    @property
    def empty(self):
        return self.frame.empty

    def __getitem__(self, item):
        return self.frame[item]

    def __repr__(self):
        result = f'{self.__class__.__qualname__}:\n\n'
        result += self.frame.__repr__()
        return result

    def to_parquet(self, path):
        """Save FrameWrapper to parquet file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.frame.to_parquet(path)

    @classmethod
    def from_parquet(cls, path) -> Self:
        """Load FrameWrapper from parquet file."""
        frame = pd.read_parquet(path)
        return cls.from_frame(frame)
