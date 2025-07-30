from __future__ import annotations

import copy

import pandas as pd
from ..grid import tile

if False:
    from .ingrid import InGrid


def __get__(
        self: SegTile,
        instance: InGrid,
        owner: type[InGrid],
) -> SegTile:
    self.ingrid = instance
    # return self.copy()
    return copy.copy(self)


class SegTile(

):
    ingrid: InGrid
    locals().update(
        __get__=__get__,
    )

    @property
    def grid(self):
        return self.ingrid

    # @cached_property
    @property
    def length(self):
        """Number of static tiles in one dimension of the segtile"""
        return self.ingrid.seggrid.tile.length

    @tile.static.cached_prpoerty
    def index(self):
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the seggrid segtile"""
        key = 'segtile.xtile'
        ingrid = self.ingrid
        if key in ingrid.columns:
            return ingrid[key]

        seggrid = ingrid.seggrid.padded
        ingrid.seggrid
        result = ingrid.xtile // ingrid.segtile.length

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        loc = ~result.isin(seggrid.xtile)

        result.isin(seggrid.xtile).any()
        result.isin(seggrid.ytile)
        assert result.isin(seggrid.xtile).all(), msg
        result.min() in seggrid.xtile.values
        import numpy as np
        np.unique(result[~result.isin(seggrid.xtile)] )
        ingrid[key] = result
        return result

    @property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the seggrid segtile"""
        key = 'segtile.ytile'
        ingrid = self.ingrid
        if key in ingrid.columns:
            return ingrid[key]

        seggrid = ingrid.seggrid.padded
        result = ingrid.ytile // ingrid.segtile.length

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(seggrid.ytile).all(), msg
        ingrid[key] = result
        return result

    @property
    def r(self) -> pd.Series:
        """row within the segtile of this tile"""
        ingrid = self.ingrid
        key = 'segtile.r'
        if key in ingrid.columns:
            return ingrid[key]
        result = (
            ingrid.ytile
            .to_series(index=ingrid.index)
            .floordiv(ingrid.segtile.length)
            .mul(ingrid.segtile.length)
            .rsub(ingrid.ytile.values)
        )
        ingrid[key] = result
        result = ingrid[key]
        return result

    @property
    def c(self) -> pd.Series:
        """column within the segtile of this tile"""
        ingrid = self.ingrid
        key = 'segtile.c'
        if key in ingrid.columns:
            return ingrid[key]
        result = (
            ingrid.xtile
            .to_series(index=ingrid.index)
            .floordiv(ingrid.segtile.length)
            .mul(ingrid.segtile.length)
            .rsub(ingrid.xtile.values)
        )
        ingrid[key] = result
        result = ingrid[key]
        return result

    @property
    def ipred(self) -> pd.Series:
        ingrid = self.ingrid4
        key = 'segtile.ipred'
        if key in ingrid.columns:
            return ingrid[key]
        arrays = self.xtile, self.ytile
        loc = pd.MultiIndex.from_arrays(arrays)
        result = (
            ingrid.seggrid.padded.ipred
            .loc[loc]
            .values
        )
        ingrid[key] = result
        return ingrid[key]

    @property
    def infile(self) -> pd.Series:
        """seggrid.file broadcasted to ingrid"""
        ingrid = self.ingrid
        key = 'segtile.infile'
        if key in ingrid.columns:
            return ingrid[key]
        result = (
            # ingrid.seggrid.file.stitched
            ingrid.seggrid.padded.file.infile
            .loc[self.index]
            .values
        )
        ingrid[key] = result
        return ingrid[key]

    def __init__(self, *args):
        ...

    def __set_name__(self, owner, name):
        self.__name__ = name
