from __future__ import annotations

import copy

import pandas as pd
from ..grid import tile

if False:
    from .ingrid import InGrid


def __get__(
        self: VecTile,
        instance: InGrid,
        owner: type[InGrid],
) -> VecTile:
    self.ingrid = instance
    return copy.copy(self)


class VecTile(

):
    ingrid: InGrid
    locals().update(
        __get__=__get__,
    )

    def __init__(self, *args):
        ...

    def __set_name__(self, owner, name):
        self.__name__ = name

    @property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the vecgrid vectile"""
        key = 'vectile.xtile'
        ingrid = self.ingrid
        if key in ingrid.columns:
            return ingrid[key]

        vecgrid = ingrid.vecgrid
        length = 2 ** (ingrid.scale - vecgrid.scale)
        result = ingrid.xtile // length

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(vecgrid.xtile).all(),msg

        ingrid[key] = result
        return result

    @property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the vecgrid vectile"""
        key = 'vectile.ytile'
        ingrid = self.ingrid
        if key in ingrid.columns:
            return ingrid[key]

        vecgrid = ingrid.vecgrid
        length = 2 ** (ingrid.scale - vecgrid.scale)
        result = ingrid.ytile // length

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(vecgrid.ytile).all(), msg

        ingrid[key] = result
        return result

    @tile.static.cached_prpoerty
    def index(self):
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @property
    def length(self):
        """Number of static grid in one dimension of the vectile"""
        return self.ingrid.vecgrid.tile.length
