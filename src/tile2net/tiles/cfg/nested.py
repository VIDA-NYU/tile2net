
from __future__ import annotations

import builtins
import functools
import inspect
from functools import *
from torch.fx.experimental.recording import trace_shape_events_log
from typing import *

from pandas.core.indexing import is_nested_tuple

if False:
    from .cfg import Cfg
    from ..tiles import Tiles


def __get__(
        self: Nested,
        instance: Nested,
        owner: Type[Nested],
):
    """"""
    from .cfg import Cfg
    self.instance = instance
    self.owner = owner
    if issubclass(owner, Cfg):
        self.cfg = instance
        self.Cfg = owner
    else:
        self.cfg = instance.cfg
        self.Cfg = instance.Cfg

    return self

class Nested(

):
    instance: Nested = None
    owner: Type[Nested] = None
    _nested: dict[str, Nested]
    cfg: Cfg = None
    Cfg: type[Cfg] = None

    locals().update(
        __get__=__get__
    )

    @cached_property
    def tiles(self) -> Tiles:
        return None

    @cached_property
    def cfg(self) -> Cfg:
        return None

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    @cached_property
    def _trace(self):
        if (
                self.instance is None
                or self.instance.instance is None
        ):
            return self.__name__
        elif isinstance(self.instance, Nested):
            return f'{self.instance._trace}.{self.__name__}'
        else:
            msg = (
                f'Cannot determine trace for {self.__name__} in '
                f'{self.owner} with {self.instance=}'
            )
            raise ValueError(msg)

    def __set_name__(
            self,
            owner: type[Nested],
            name
    ):
        self.__name__ = name
        self.owner = owner
        if issubclass(owner, Nested):
            if '_nested' not in owner.__dict__:
                owner._nested = {}
            owner._nested[name] = self


