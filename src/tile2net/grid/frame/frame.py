from __future__ import annotations
from .framewrapper import FrameWrapper

import copy
from functools import *
from typing import *
from tile2net.grid.frame.namespace import namespace
import pandas as pd

if False:
    from .framewrapper import FrameWrapper


class Column(
    namespace
):
    """
    Wraps column operations (get, set, del) for Grid.grid
    """
    test = None

    def __set_name__(self, owner, name):
        self.__name__ = name

    def _get(
            self,
            instance: FrameWrapper,
            owner: type
    ) -> Self | pd.Series:
        self = super()._get(instance, owner)
        frame = self.wrapper.frame
        result = copy.copy(self)
        if instance is None:
            return result
        key = self.key
        if key in frame:
            return frame[key]
        if key in frame.index.names:
            return frame.index.get_level_values(key)
        wrapped = self.__wrapped__
        if wrapped is None:
            msg = (
                f'No wrapper implemented for '
                f'{owner.__qualname__}.{self.__name__}'
            )
            raise NotImplementedError(msg)
        result = (
            self.__wrapped__
            .__get__(instance, owner)
            .__call__()
        )
        frame[key] = result
        result = frame[key]
        return result

    locals().update(
        __get__=_get
    )

    if False:
        def __get__(self, instance, owner) -> Union[
            Self,
            pd.Series,
            Iterable
        ]:
            ...

    @cached_property
    def key(self):
        instance = self
        names = []
        while True:
            names.append(instance.__name__)
            instance = instance.instance
            if (
                    instance is None
                    or isinstance(instance, FrameWrapper)
            ):
                break

        result = '.'.join(names[::-1])
        return result

    def __set__(
            self,
            instance: namespace,
            value,
    ):
        self.instance = instance
        self.wrapper = instance.wrapper
        wrapper: FrameWrapper = self.wrapper
        cache = wrapper.frame.__dict__
        key = self.key
        cache[key] = value

    def __delete__(
            self,
            instance: namespace,
    ):
        self.instance = instance
        self.wrapper = instance.wrapper
        wrapper: FrameWrapper = self.wrapper
        cache = wrapper.frame.__dict__
        key = self.key
        del cache[key]


class Index(Column):
    # def __get__(
    #         self,
    #         instance,
    #         owner
    # ) -> Union[
    #     Self,
    #     pd.Index,
    #     pd.Series,
    # ]:
    #     self = super()._get(instance, owner)
    #     frame = self.frame
    #     try:
    #         return frame.index.get_level_values(self.key)
    #     except KeyError:
    #         return super().__get__(instance, owner)
    ...



def column(
        *args, **kwargs
) -> Union[
    pd.Series,
    Column
]:
    return Column(*args, **kwargs)


def index(
        *args, **kwargs
) -> Union[
    pd.Index,
    Index
]:
    return Index(*args, **kwargs)
