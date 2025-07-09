from __future__ import annotations
import pandas as pd

import copy

import pandas as pd
from ..tiles import tile

if False:
    from .segtiles import SegTiles


def __get__(
        self: VecTile,
        instance: SegTiles,
        owner: type[SegTiles],
) -> VecTile:
    self.segtiles = instance
    return copy.copy(self)


class VecTile(

):
    segtiles: SegTiles
    locals().update(
        __get__=__get__,
    )

    @property
    def tiles(self):
        return self.segtiles

    @property
    def vectiles(self):
        return self.segtiles.vectiles

    @property
    def intiles(self):
        return self.segtiles.intiles

    @property
    def length(self):
        """Number of tiles in one dimension of the vectile"""
        return self.vectiles.tile.length

    @property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the vectiles vectile"""
        key = 'vectile.xtile'
        segtiles = self.segtiles
        if key in segtiles.columns:
            return segtiles[key]

        vectiles = segtiles.vectiles
        length = 2 ** (segtiles.tile.scale - vectiles.tile.scale)
        result = segtiles.xtile // length

        msg = 'All segtile.xtile must be in segtiles.xtile!'
        assert result.isin(vectiles.xtile).all(),msg

        segtiles[key] = result
        return result

    @property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the vectiles vectile"""
        key = 'vectile.ytile'
        segtiles = self.segtiles
        if key in segtiles.columns:
            return segtiles[key]

        vectiles = segtiles.vectiles
        length = 2 ** (segtiles.tile.scale - vectiles.tile.scale)
        result = segtiles.ytile // length

        msg = 'All segtile.ytile must be in segtiles.ytile!'
        assert result.isin(vectiles.ytile).all(), msg

        segtiles[key] = result
        return result

    # @tile.cached_property
    # def frame(self) -> pd.DataFrame:
    #     raise NotImplementedError
    #     segtiles = self.segtiles
    #     concat = []
    #     ytile = segtiles.ytile // segtiles.vectile.length
    #     xtile = segtiles.xtile // segtiles.vectile.length
    #     frame = pd.DataFrame(dict(
    #         xtile=xtile,
    #         ytile=ytile,
    #     ), index=segtiles.index)
    #     concat.append(frame)
    #


    # @property
    # def group(self) -> pd.Series:
    #     segtiles = self.segtiles
    #     key = 'vectile.group'
    #     if key in segtiles.columns:
    #         return segtiles[key]
    #     arrays = self.xtile, self.ytile
    #     loc = pd.MultiIndex.from_arrays(arrays)
    #     result = (
    #         segtiles.vectiles.ipred
    #         .loc[loc]
    #         .values
    #     )
    #     segtiles[key] = result
    #     return segtiles[key]

    @property
    def grayscale(self) -> pd.Series:
        """segtiles.file broadcasted to intiles"""
        segtiles = self.segtiles
        key = 'segtile.grayscale'
        if key in segtiles.columns:
            return segtiles[key]
        vectiles = self.vectiles
        result = (
            vectiles.file.grayscale
            .loc[self.index]
            .values
        )
        segtiles[key] = result
        return segtiles[key]

    @property
    def infile(self) -> pd.Series:
        """segtiles.file broadcasted to intiles"""
        segtiles = self.segtiles
        key = 'segtile.infile'
        if key in segtiles.columns:
            return segtiles[key]
        vectiles = self.vectiles
        result = (
            vectiles.file.infile
            .loc[self.index]
            .values
        )
        segtiles[key] = result
        return segtiles[key]

    @property
    def colored(self) -> pd.Series:
        """segtiles.file broadcasted to intiles"""
        segtiles = self.segtiles
        key = 'segtile.colored'
        if key in segtiles.columns:
            return segtiles[key]
        vectiles = self.vectiles
        result = (
            vectiles.file.colored
            .loc[self.index]
            .values
        )
        segtiles[key] = result
        return segtiles[key]


    @tile.static.cached_prpoerty
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
