from __future__ import annotations
from .. import frame

import copy

import pandas as pd

from tile2net.grid.frame.namespace import namespace

if False:
    from .ingrid import InGrid


class SegTile(
    namespace,
):
    ingrid: InGrid

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

    @frame.column
    def r(self) -> pd.Series:
        """row within the segtile of this tile"""
        ingrid = self.ingrid
        result = (
            ingrid.ytile
            .to_series(index=ingrid.index)
            .floordiv(ingrid.segtile.length)
            .mul(ingrid.segtile.length)
            .rsub(ingrid.ytile.values)
        )
        return result

    @frame.column
    def c(self) -> pd.Series:
        """column within the segtile of this tile"""
        ingrid = self.ingrid
        result = (
            ingrid.xtile
            .to_series(index=ingrid.index)
            .floordiv(ingrid.segtile.length)
            .mul(ingrid.segtile.length)
            .rsub(ingrid.xtile.values)
        )
        return result

    @frame.column
    def infile(self) -> pd.Series:
        """seggrid.file broadcasted to ingrid"""
        ingrid = self.ingrid
        result = (
            # ingrid.seggrid.file.stitched
            ingrid.seggrid.padded.file.infile
            .loc[self.index]
            .values
        )
        return result
