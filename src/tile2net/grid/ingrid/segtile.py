from __future__ import annotations
from .. import frame

import copy

import pandas as pd

from tile2net.grid.frame.namespace import namespace

if False:
    from .ingrid import InGrid


def __get__(
        self: SegTile,
        instance: InGrid,
        owner: type[InGrid],
) -> SegTile:
    self.ingrid = instance
    return copy.copy(self)


class SegTile(
    namespace,
):
    ingrid: InGrid

    # locals().update(
    #     __get__=__get__,
    # )

    @property
    def ingrid(self) -> InGrid:
        return self.instance

    @property
    def grid(self) -> InGrid:
        return self.ingrid

    @property
    def length(self):
        """Number of static grid in one dimension of the segtile"""
        return self.ingrid.seggrid.length

    @property
    def index(self):
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @frame.column
    def xtile(self):
        """Tile integer X of this tile in the seggrid segtile"""
        ingrid = self.ingrid

        seggrid = ingrid.seggrid.padded
        result: pd.Index = ingrid.xtile.__floordiv__(ingrid.segtile.length)

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(seggrid.xtile).all(), msg
        return result

    @frame.column
    def ytile(self):
        """Tile integer Y of this tile in the seggrid segtile"""
        ingrid = self.ingrid

        seggrid = ingrid.seggrid.padded
        result: pd.Index = ingrid.ytile.__floordiv__(ingrid.segtile.length)

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(seggrid.ytile).all(), msg
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
        ingrid = self.ingrid
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
