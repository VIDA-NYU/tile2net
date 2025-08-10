from __future__ import annotations

import copy
from functools import *
from typing import *

import pandas as pd
from geopandas import GeoDataFrame

from tile2net.grid.util import returns_or_assigns

if False:
    from tile2net.grid.grid.grid import Grid

TGrid = TypeVar('TGrid', covariant=True)


class namespace(
    Generic[TGrid]
):
    instance: TGrid = None
    frame: GeoDataFrame = None
    __wrapped__ = None
    __name__ = None

    def _get(
            self,
            instance: TGrid,
            owner
    ) :
        self.instance = instance
        self.frame = instance.frame
        return copy.copy(self)

    locals().update(
        __get__=_get,
    )

    def __init__(
            self,
            func=None,
            *args,
            **kwargs
    ):

        if returns_or_assigns(func):
            update_wrapper(self, func)

    def __set_name__(self, owner, name):
        self.__name__ = name
