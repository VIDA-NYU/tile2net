from __future__ import annotations

import copy

import pandas as pd
from ..tiles import tile

if False:
    from .intiles import InTiles


def __get__(
        self: SegTile,
        instance: InTiles,
        owner: type[InTiles],
) -> SegTile:
    self.intiles = instance
    # return self.copy()
    return copy.copy(self)


class SegTile(

):
    intiles: InTiles
    locals().update(
        __get__=__get__,
    )

    @property
    def tiles(self):
        return self.intiles

    # @tile.cached_property
    @property
    def length(self):
        """Number of static tiles in one dimension of the segtile"""
        return self.intiles.segtiles.tile.length

    @tile.static.cached_prpoerty
    def index(self):
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the segtiles segtile"""
        key = 'segtile.xtile'
        intiles = self.intiles
        if key in intiles.columns:
            return intiles[key]

        segtiles = intiles.segtiles.padded
        # unpadded = inti.unpadded.index.isin(segtiles.index)
        intiles.segtiles
        result = intiles.xtile // intiles.segtile.length

        msg = 'All segtile.xtile must be in segtiles.xtile!'
        loc = ~result.isin(segtiles.xtile)

        result.isin(segtiles.xtile).any()
        result.isin(segtiles.ytile)
        assert result.isin(segtiles.xtile).all(), msg
        result.min() in segtiles.xtile.values
        import numpy as np
        np.unique(result[~result.isin(segtiles.xtile)] )
        intiles[key] = result
        return result

    @property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the segtiles segtile"""
        key = 'segtile.ytile'
        intiles = self.intiles
        if key in intiles.columns:
            return intiles[key]

        segtiles = intiles.segtiles.padded
        result = intiles.ytile // intiles.segtile.length

        msg = 'All segtile.ytile must be in segtiles.ytile!'
        assert result.isin(segtiles.ytile).all(), msg
        intiles[key] = result
        return result

    @property
    def r(self) -> pd.Series:
        """row within the segtile of this tile"""
        intiles = self.intiles
        key = 'segtile.r'
        if key in intiles.columns:
            return intiles[key]
        result = (
            intiles.ytile
            .to_series(index=intiles.index)
            .floordiv(intiles.segtile.length)
            .mul(intiles.segtile.length)
            .rsub(intiles.ytile.values)
        )
        intiles[key] = result
        result = intiles[key]
        return result

    @property
    def c(self) -> pd.Series:
        """column within the segtile of this tile"""
        intiles = self.intiles
        key = 'segtile.c'
        if key in intiles.columns:
            return intiles[key]
        result = (
            intiles.xtile
            .to_series(index=intiles.index)
            .floordiv(intiles.segtile.length)
            .mul(intiles.segtile.length)
            .rsub(intiles.xtile.values)
        )
        intiles[key] = result
        result = intiles[key]
        return result

    @property
    def ipred(self) -> pd.Series:
        intiles = self.intiles
        key = 'segtile.ipred'
        if key in intiles.columns:
            return intiles[key]
        arrays = self.xtile, self.ytile
        loc = pd.MultiIndex.from_arrays(arrays)
        result = (
            intiles.segtiles.padded.ipred
            .loc[loc]
            .values
        )
        intiles[key] = result
        return intiles[key]

    @property
    def stitched(self) -> pd.Series:
        """segtiles.file broadcasted to intiles"""
        intiles = self.intiles
        key = 'segtile.file'
        if key in intiles.columns:
            return intiles[key]
        result = (
            # intiles.segtiles.file.stitched
            intiles.segtiles.padded.file.stitched
            .loc[self.index]
            .values
        )
        intiles[key] = result
        return intiles[key]

    def __init__(self, *args):
        ...

    def __set_name__(self, owner, name):
        self.__name__ = name
