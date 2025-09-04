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
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension, self.ingrid.shape[2]

    @property
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension,

    @frame.column
    def xtile(self):
        """Tile integer X of this tile in the vecgrid vectile"""
        seggrid = self.seggrid
        vecgrid = seggrid.vecgrid
        length = 2 ** (seggrid.scale - vecgrid.scale)
        result = seggrid.xtile // length

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(vecgrid.xtile).all(), msg
        return result

    @frame.column
    def ytile(self):
        """Tile integer X of this tile in the vecgrid vectile"""
        seggrid = self.seggrid
        vecgrid = seggrid.vecgrid
        length = 2 ** (seggrid.scale - vecgrid.scale)
        result = seggrid.ytile // length

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(vecgrid.ytile).all(), msg

        return result

    @frame.column
    def itile(self):
        """Integer identifier for each segtile"""
        result = (
            self.grid.vecgrid.itile
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def grayscale(self):
        """seggrid.file broadcasted to ingrid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.grayscale
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def infile(self):
        """seggrid.file broadcasted to ingrid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.infile
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def colored(self):
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

    @frame.column
    def polygon(self):
        result = (
            self.grid.vecgrid.file.polygons
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def line(self):
        result = (
            self.grid.vecgrid.file.lines
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def affine(self):
        vecgrid = self.vecgrid
        result = (
            vecgrid.affine_params
            .loc[self.index]
            .values
        )
        return result

    # west/east/south/north bounds in projected coords (broadcast)
    @frame.column
    def lonmin(self):
        result = (
            self.vecgrid.lonmin
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def latmin(self):
        result = (
            self.vecgrid.latmin
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def lonmax(self):
        result = (
            self.vecgrid.lonmax
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def latmax(self):
        result = (
            self.vecgrid.latmax
            .loc[self.index]
            .values
        )
        return result

    # output file paths for polygons and lines (broadcast)
    @frame.column
    def polygon_file(self):
        result = (
            self.grid.vecgrid.file.polygons
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def line_file(self):
        result = (
            self.grid.vecgrid.file.lines
            .loc[self.index]
            .values
        )
        return result
