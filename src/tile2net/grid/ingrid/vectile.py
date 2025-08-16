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
        return self.instance

    @frame.column
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the vecgrid vectile"""
        ingrid = self.ingrid

        vecgrid = ingrid.vecgrid
        length = 2 ** (ingrid.scale - vecgrid.scale)
        result = ingrid.xtile // length

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(vecgrid.xtile).all(),msg
        return result

    @frame.column
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the vecgrid vectile"""
        ingrid = self.ingrid

        vecgrid = ingrid.vecgrid
        length = 2 ** (ingrid.scale - vecgrid.scale)
        result = ingrid.ytile // length

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(vecgrid.ytile).all(), msg
        return result

    @property
    def index(self) -> pd.MultiIndex:
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @property
    def length(self):
        """Number of static grid in one dimension of the vectile"""
        return self.ingrid.vecgrid.length
