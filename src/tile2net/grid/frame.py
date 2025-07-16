from __future__ import annotations

import copy
from functools import *
from typing import *

import pandas as pd
from geopandas import GeoDataFrame

from .namespace import namespace
from .util import returns_or_assigns

if False:
    from .framewrapper import FrameWrapper


class column(namespace):
    """
    Wraps column operations (get, set, del) for Grid.tiles
    """
    test = None

    def __init__(self, func):
        if returns_or_assigns(func):
            update_wrapper(self, func)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(
            self,
            instance: FrameWrapper,
            owner
    ) -> Self | pd.Series:
        self.instance = instance
        self.frame = instance.frame
        frame = self.tiles
        result = copy.copy(self)
        if instance is None:
            return result
        key = self.key
        if key in frame:
            return frame[key]
        wrapped = self.__wrapped__
        if wrapped is None:
            msg = f'No wrapper implemented for {owner}.{self.__name__}'
            raise NotImplementedError(msg)
        result = (
            self.__wrapped__
            .__get__(instance, owner)
        )
        frame[key] = result
        result = frame[key]
        return result

    # locals().update(
    #     __get__=__get
    # )

    def __set__(
            self,
            instance,
            value,
    ):
        self.instance = instance
        self.frame = frame = instance.frame
        result = copy.copy(self)
        if instance is None:
            return result
        key = self.key
        frame[key] = value

    @cached_property
    def key(self):
        from .grid import Grid
        instance = self.instance
        names = []
        while not isinstance(instance, Grid):
            names.append(instance.__name__)
            instance = instance.instance
        result = '.'.join(names[::-1])
        return result

    def __delete__(
            self,
            instance,
    ):
        self.instance = instance
        self.frame = frame = instance.frame
        result = copy.copy(self)
        if instance is None:
            return result
        key = self.key
        try:
            del frame[key]
        except KeyError:
            ...


class index(column):
    def __get__(
            self,
            instance,
            owner
    ) -> Union[
        Self,
        pd.Index,
        pd.Series,
    ]:
        self.instance = instance
        self.frame = instance.frame
        frame = self.frame
        try:
            return frame.index.get_level_values(self.key)
        except KeyError:
            return super().__get__(instance, owner)
