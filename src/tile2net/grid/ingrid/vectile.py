from __future__ import annotations
from .. import frame

import copy

import pandas as pd

from tile2net.grid.frame.namespace import namespace

if False:
    from .ingrid import InGrid



class VecTile(
    namespace,
):

    ingrid: InGrid

    @property
    def ingrid(self):
        """Reference to parent InGrid instance."""
        return self.instance

    @frame.column
    def xtile(self) -> pd.Series:
        """
        X coordinate of vectorization tile in vecgrid space.

        X tile id of the VecGrid tile associated with the InGrid tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vectile.xtile
            xtile   ytile
            317280  387840    9915
            Name: vectile.xtile, dtype: int64
        """
        ingrid = self.ingrid

        vecgrid = ingrid.vecgrid
        length = 2 ** (ingrid.scale - vecgrid.scale)
        result = ingrid.xtile // length

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(vecgrid.xtile).all(),msg
        return result

    @frame.column
    def ytile(self) -> pd.Series:
        """
        Y coordinate of vectorization tile in vecgrid space.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vectile.ytile
            xtile   ytile
            317280  387840    12120
            Name: vectile.ytile, dtype: int64
        """
        ingrid = self.ingrid

        vecgrid = ingrid.vecgrid
        length = 2 ** (ingrid.scale - vecgrid.scale)
        result = ingrid.ytile // length

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(vecgrid.ytile).all(), msg
        return result

    @property
    def index(self) -> pd.MultiIndex:
        """
        MultiIndex of (xtile, ytile) for vectorization tiles.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vectile.index
            MultiIndex([(9915, 12120)], names=['vectile.xtile', 'vectile.ytile'])
        """
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @property
    def length(self):
        """
        Number of InGrid tiles in one dimension of the vectorization tile.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vectile.length
            32
        """
        return self.ingrid.vecgrid.length
