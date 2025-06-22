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

    # @tile.cached_property
    @property
    def length(self):
        """Number of tiles in one dimension of the vectile"""
        return self.vectiles.tile.length

    @tile.cached_property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the vectiles vectile"""
        raise NotImplementedError

    @tile.cached_property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the vectiles vectile"""
        raise NotImplementedError
        # todo: for each vectile origin, use array ranges to pad the tiles

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
    def stitched(self) -> pd.Series:
        """segtiles.file broadcasted to intiles"""
        segtiles = self.segtiles
        key = 'segtile.file'
        if key in segtiles.columns:
            return segtiles[key]
        vectiles = self.vectiles
        result = (
            vectiles.file.stitched
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
