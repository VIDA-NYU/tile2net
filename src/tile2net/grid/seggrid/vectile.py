from __future__ import annotations
from tile2net.grid.frame.namespace import namespace
import pandas as pd

import copy

import pandas as pd
from ..grid import grid

if False:
    from .seggrid import SegGrid


def __get__(
        self: VecTile,
        instance: SegGrid,
        owner: type[SegGrid],
) -> VecTile:
    self.seggrid = instance
    return copy.copy(self)


class VecTile(
    namespace,
):
    seggrid: SegGrid
    # locals().update(
    #     __get__=__get__,
    # )

    @property
    def seggrid(self):
        return self.instance

    @property
    def grid(self):
        return self.seggrid

    @property
    def vecgrid(self):
        return self.seggrid.vecgrid

    @property
    def ingrid(self):
        return self.seggrid.ingrid

    @property
    def length(self):
        """Number of grid in one dimension of the vectile"""
        return self.vecgrid.length

    @property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the vecgrid vectile"""
        key = 'vectile.xtile'
        seggrid = self.seggrid
        if key in seggrid.frame.columns:
            return seggrid[key]

        vecgrid = seggrid.vecgrid
        length = 2 ** (seggrid.scale - vecgrid.scale)
        result = seggrid.xtile // length

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(vecgrid.xtile).all(),msg

        seggrid[key] = result
        return result

    @property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the vecgrid vectile"""
        key = 'vectile.ytile'
        seggrid = self.seggrid
        if key in seggrid.frame.columns:
            return seggrid[key]

        vecgrid = seggrid.vecgrid
        length = 2 ** (seggrid.scale - vecgrid.scale)
        result = seggrid.ytile // length

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(vecgrid.ytile).all(), msg

        seggrid[key] = result
        return result

    # @cached_property
    # def frame(self) -> pd.DataFrame:
    #     raise NotImplementedError
    #     seggrid = self.seggrid
    #     concat = []
    #     ytile = seggrid.ytile // seggrid.vectile.length
    #     xtile = seggrid.xtile // seggrid.vectile.length
    #     frame = pd.DataFrame(dict(
    #         xtile=xtile,
    #         ytile=ytile,
    #     ), index=seggrid.index)
    #     concat.append(frame)
    #


    # @property
    # def group(self) -> pd.Series:
    #     seggrid = self.seggrid
    #     key = 'vectile.group'
    #     if key in seggrid.columns:
    #         return seggrid[key]
    #     arrays = self.xtile, self.ytile
    #     loc = pd.MultiIndex.from_arrays(arrays)
    #     result = (
    #         seggrid.vecgrid.ipred
    #         .loc[loc]
    #         .values
    #     )
    #     seggrid[key] = result
    #     return seggrid[key]

    @property
    def grayscale(self) -> pd.Series:
        """seggrid.file broadcasted to ingrid"""
        seggrid = self.seggrid
        key = 'segtile.grayscale'
        if key in seggrid.columns:
            return seggrid[key]
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.grayscale
            .loc[self.index]
            .values
        )
        seggrid[key] = result
        return seggrid[key]

    @property
    def infile(self) -> pd.Series:
        """seggrid.file broadcasted to ingrid"""
        seggrid = self.seggrid
        key = 'segtile.infile'
        if key in seggrid.columns:
            return seggrid[key]
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.infile
            .loc[self.index]
            .values
        )
        seggrid[key] = result
        return seggrid[key]

    @property
    def colored(self) -> pd.Series:
        """seggrid.file broadcasted to ingrid"""
        seggrid = self.seggrid
        key = 'segtile.colored'
        if key in seggrid.columns:
            return seggrid[key]
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.colored
            .loc[self.index]
            .values
        )
        seggrid[key] = result
        return seggrid[key]


    @property
    def index(self):
        # todo: we need sticky (attrs) and not-stick (__dict__)
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    def __init__( self, *args ):
        ...

    def __set_name__(self, owner, name):
        self.__name__ = name
