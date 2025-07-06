from __future__ import annotations

import copy

import pandas as pd
from ..tiles import tile

if False:
    from .intiles import InTiles


def __get__(
        self: VecTile,
        instance: InTiles,
        owner: type[InTiles],
) -> VecTile:
    self.intiles = instance
    return copy.copy(self)


class VecTile(

):
    intiles: InTiles
    locals().update(
        __get__=__get__,
    )

    def __init__(self, *args):
        ...

    def __set_name__(self, owner, name):
        self.__name__ = name

    @property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the vectiles vectile"""
        key = 'vectile.xtile'
        intiles = self.intiles
        if key in intiles.columns:
            return intiles[key]

        vectiles = intiles.vectiles
        length = 2 ** (intiles.tile.scale - vectiles.tile.scale)
        result = intiles.xtile // length

        msg = 'All segtile.xtile must be in segtiles.xtile!'
        assert result.isin(vectiles.xtile).all(),msg

        intiles[key] = result
        return result

    @property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the vectiles vectile"""
        key = 'vectile.ytile'
        intiles = self.intiles
        if key in intiles.columns:
            return intiles[key]

        vectiles = intiles.vectiles
        length = 2 ** (intiles.tile.scale - vectiles.tile.scale)
        result = intiles.ytile // length

        msg = 'All segtile.ytile must be in segtiles.ytile!'
        assert result.isin(vectiles.ytile).all(), msg

        intiles[key] = result
        return result

    @tile.static.cached_prpoerty
    def index(self):
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @property
    def length(self):
        """Number of static tiles in one dimension of the vectile"""
        return self.intiles.vectiles.tile.length
