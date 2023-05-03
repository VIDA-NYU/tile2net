from __future__ import annotations
from typing import Callable, TypeVar

from functools import cached_property
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame

from pandas.core.indexing import _LocIndexer, _iLocIndexer


__all__ = ['Frame']

F = TypeVar('F', bound='Frame')

class iLocIndexer(_iLocIndexer):
    __getitem__: Callable[..., F]

class LocIndexer(_LocIndexer):
    __getitem__: Callable[..., F]

class FrameMeta(type):
    ...

class Frame(DataFrame, metaclass=FrameMeta):
    @cached_property
    def _constructor(self):
        return type(self)

    @property
    def loc(self) -> LocIndexer:
        return LocIndexer('loc', self)

    @property
    def iloc(self) -> iLocIndexer:
        return iLocIndexer('iloc', self)

    def __hash__(self):
        # to support caching
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if isinstance(data, NDFrame):
            self.attrs = data.attrs.copy()

    def __repr__(self):
        constructor = self._constructor
        self._constructor = self.__class__.__base__
        res = super().__repr__()
        self._constructor = constructor
        return res

def mro(cls: FrameMeta) -> list[type]:
    # prioritize attributes explicitly defined in frame
    mro = type.mro(cls)
    mro.remove(Frame)
    mro.insert(1, Frame)
    return mro

FrameMeta.mro = mro
