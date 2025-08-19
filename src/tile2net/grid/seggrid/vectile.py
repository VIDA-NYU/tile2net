from __future__ import annotations
from tile2net.grid.frame.namespace import namespace
from .. import frame

import copy

import pandas as pd
from ..grid import grid

if False:
    from .seggrid import SegGrid
    from ..vecgrid.vecgrid import VecGrid
    from ..ingrid.ingrid import InGrid


class VecTile(
    namespace,
):
    seggrid: SegGrid

    @property
    def seggrid(self) -> SegGrid:
        return self.instance

    @property
    def grid(self):
        return self.seggrid

    @property
    def vecgrid(self) -> VecGrid:
        return self.seggrid.vecgrid

    @property
    def ingrid(self) -> InGrid:
        return self.seggrid.ingrid

    # @property
    # def length(self):
    #     """Number of grid in one dimension of the vectile"""
    #     return self.vecgrid.length

    @property
    def length(self) -> int:
        return self.vecgrid.length

    @property
    def dimension(self):
        return self.length * self.seggrid.dimension

    @property
    def shape(self) -> tuple[int,int]:
        return self.dimension, self.dimension, self.ingrid.shape[2]

    @property
    def shape(self) -> tuple[int,int]:
        return self.dimension, self.dimension,

    @frame.column
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the vecgrid vectile"""
        seggrid = self.seggrid
        vecgrid = seggrid.vecgrid
        length = 2 ** (seggrid.scale - vecgrid.scale)
        result = seggrid.xtile // length

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(vecgrid.xtile).all(),msg
        return result

    @frame.column
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the vecgrid vectile"""
        seggrid = self.seggrid
        vecgrid = seggrid.vecgrid
        length = 2 ** (seggrid.scale - vecgrid.scale)
        result = seggrid.ytile // length

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(vecgrid.ytile).all(), msg
        
        return result

    @frame.column
    def grayscale(self) -> pd.Series:
        """seggrid.file broadcasted to ingrid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.grayscale
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def infile(self) -> pd.Series:
        """seggrid.file broadcasted to ingrid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.infile
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def colored(self) -> pd.Series:
        """seggrid.file broadcasted to ingrid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.colored
            .loc[self.index]
            .values
        )
        return result


    @property
    def index(self):
        # todo: we need sticky (attrs) and not-stick (__dict__)
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @frame.column
    def r(self):
        """row within the segtile of this tile"""

        ytile = self.grid.ytile.to_series()
        result = (
            ytile
            .groupby(self.ytile.values)
            .min()
            .loc[self.ytile]
            .rsub(ytile.values)
            .values
        )
        return result

    @frame.column
    def c(self):
        """column within the segtile of this tile"""
        xtile = self.grid.xtile.to_series()
        result = (
            xtile
            .groupby(self.xtile.values)
            .min()
            .loc[self.xtile]
            .rsub(xtile.values)
            .values
        )
        return result

    # @frame.column
    # def r(self) -> pd.Series:
    #     """row within the segtile of this tile"""
    #     ingrid = self.seggrid
    #     result = (
    #         ingrid.ytile
    #         .to_series(index=ingrid.index)
    #         .floordiv(ingrid.vectile.length)
    #         .mul(ingrid.vectile.length)
    #         .rsub(ingrid.ytile.values)
    #     )
    #     return result
    #
    # @frame.column
    # def c(self) -> pd.Series:
    #     """column within the segtile of this tile"""
    #     ingrid = self.seggrid
    #     result = (
    #         ingrid.xtile
    #         .to_series(index=ingrid.index)
    #         .floordiv(ingrid.vectile.length)
    #         .mul(ingrid.vectile.length)
    #         .rsub(ingrid.xtile.values)
    #     )
    #     return result




