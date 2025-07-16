from __future__ import annotations

from __future__ import annotations

import copy
from typing import *

from tile2net.grid.namespace import namespace

if False:
    from tile2net.grid.grid.grid import Grid


class Static(namespace):
    def __get__(
            self,
            instance: Grid,
            owner
    ) -> dict | Self:
        self.instance = instance
        self.tiles = tiles = instance.tiles
        result = copy.copy(self)
        if instance is None:
            return result
        result = instance.__dict__.setdefault(self.__name__, {})
        return result

    def __delete__(
            self,
            instance: Grid,
    ):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            ...


class cached_property(namespace):
    ...
