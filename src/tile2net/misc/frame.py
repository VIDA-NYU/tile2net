from __future__ import annotations

import time
from typing import Callable, TypeVar

from functools import cached_property
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame

from pandas.core.indexing import _LocIndexer, _iLocIndexer
from tile2net.misc.attrs import attr, column

__all__ = ['Frame']

F = TypeVar('F', bound='Frame')


class iLocIndexer(_iLocIndexer):
    __getitem__: Callable[..., F]


class LocIndexer(_LocIndexer):
    __getitem__: Callable[..., F]


class FrameMeta(type):
    ...
    # @property
    # def frame_attrs(cls) -> dict[str, attr]:
    #     result: dict[str, attr] = {
    #         key: value
    #         for parent in cls.mro()
    #         if isinstance(parent, FrameMeta)
    #         for key, value in parent.__dict__.items()
    #         if isinstance(value, attr)
    #     }
    #     return result


class Frame(DataFrame, metaclass=FrameMeta):

    # class Frame(DataFrame):
    @cached_property
    def _constructor(self):
        return type(self)

    @property
    def loc(self) -> LocIndexer:
        return LocIndexer('loc', self)

    @property
    def iloc(self) -> iLocIndexer:
        return iLocIndexer('iloc', self)

    @attr
    @property
    def timestamp(self):
        return time.time().__int__()

    def __hash__(self):
        # to support caching
        return hash(self.timestamp)

    def __eq__(self, other):
        return self.timestamp == other.timestamp

    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if isinstance(data, NDFrame):
            self.attrs = data.attrs.copy()

    def __repr__(self):
        constructor = self._constructor
        self._constructor = DataFrame
        # res = super().__repr__()
        res = super(Frame, self).__repr__()
        self._constructor = constructor
        return res

    def set_axis(
            self,
            labels,
            *,
            axis: Axis = 0,
            copy: bool | None = None,
    ) -> DataFrame:
        result = super().set_axis(labels, axis=axis, copy=copy)
        return result

    def flush_columns(self):
        # del all column methods
        cls = self.__class__
        for col in self:
            if (
                    hasattr(cls, col)
                    and isinstance(getattr(cls, col), column)
            ):
                delattr(self, col)
        return self


def mro(cls: FrameMeta) -> list[type]:
    # prioritize attributes explicitly defined in frame
    mro = type.mro(cls)
    mro.remove(Frame)
    mro.insert(1, Frame)
    return mro
