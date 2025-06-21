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

    # @tile.cached_property
    @property
    def length(self):
        """Number of tiles in one dimension of the mosaic"""
        return self.segtiles.vectiles.tile.length

    @tile.cached_property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the vectiles mosaic"""
        key = 'mosaic.xtile'
        segtiles = self.segtiles
        if key in segtiles.columns:
            return segtiles[key]

        vectiles = segtiles.vectiles
        result = segtiles.xtile // segtiles.vectile.length

        msg = 'All mosaic.xtile must be in vectiles.xtile!'
        assert result.isin(vectiles.xtile).all(), msg
        segtiles[key] = result
        return result

    @tile.cached_property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the vectiles mosaic"""
        segtiles = self.segtiles
        vectiles = segtiles.vectiles
        result = segtiles.ytile // segtiles.vectile.length

        # todo: for each vectile origin, use array ranges to pad the tiles

    @tile.cached_property
    def frame(self) -> pd.DataFrame:
        segtiles = self.segtiles
        concat = []
        ytile = segtiles.ytile // segtiles.vectile.length
        xtile = segtiles.xtile // segtiles.vectile.length
        frame = pd.DataFrame(dict(
            xtile=xtile,
            ytile=ytile,
        ), index=segtiles.index)
        concat.append(frame)

    @property
    def group(self) -> pd.Series:
        segtiles = self.segtiles
        key = 'mosaic.group'
        if key in segtiles.columns:
            return segtiles[key]
        arrays = self.xtile, self.ytile
        loc = pd.MultiIndex.from_arrays(arrays)
        result = (
            segtiles.vectiles.ipred
            .loc[loc]
            .values
        )
        segtiles[key] = result
        return segtiles[key]


    def __init__(
            self,
            *args,
            **kwargs
    ):
        ...
