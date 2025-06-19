from __future__ import annotations

import pandas as pd
from ..tiles import tile

if False:
    from .intiles import InTiles


def __get__(
        self: PredTile,
        instance: InTiles,
        owner: type[InTiles],
) -> PredTile:
    self.intiles = instance
    self.InTiles = owner
    return self


class PredTile(

):
    intiles: InTiles
    locals().update(
        __get__=__get__,
    )

    @tile.cached_property
    def length(self):
        """Number of tiles in one dimension of the mosaic"""
        length = int(
            self.intiles.predtiles.tile.scale
            - self.intiles.tile.scale
        )
        return length ** 2

    @tile.cached_property
    def index(self):
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the predtiles mosaic"""
        key = 'mosaic.xtile'
        intiles = self.intiles
        if key in intiles.columns:
            return intiles[key]

        predtiles = intiles.predtiles
        result = intiles.xtile // intiles.predtile.length

        msg = 'All mosaic.xtile must be in predtiles.xtile!'
        assert result.isin(predtiles.xtile).all(), msg
        intiles[key] = result
        return result

    @property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the predtiles mosaic"""
        key = 'mosaic.ytile'
        intiles = self.intiles
        if key in intiles.columns:
            return intiles[key]

        predtiles = intiles.predtiles
        result = intiles.ytile // intiles.predtile.length

        msg = 'All mosaic.ytile must be in predtiles.ytile!'
        assert result.isin(predtiles.ytile).all(), msg
        intiles[key] = result
        return result

    @property
    def r(self) -> pd.Series:
        """row within the mosaic of this tile"""
        intiles = self.intiles
        key = 'mosaic.r'
        if key in intiles.columns:
            return intiles[key]
        result = (
            intiles.ytile
            .to_series(index=intiles.index)
            .floordiv(intiles.predtile.length)
            .mul(intiles.predtile.length)
            .rsub(intiles.ytile.values)
        )
        intiles[key] = result
        result = intiles[key]
        return result

    @property
    def c(self) -> pd.Series:
        """column within the mosaic of this tile"""
        intiles = self.intiles
        key = 'mosaic.c'
        if key in intiles.columns:
            return intiles[key]
        result = (
            intiles.xtile
            .to_series(index=intiles.index)
            .floordiv(intiles.predtile.length)
            .mul(intiles.predtile.length)
            .rsub(intiles.xtile.values)
        )
        intiles[key] = result
        result = intiles[key]
        return result

    @property
    def ipred(self) -> pd.Series:
        intiles = self.intiles
        key = 'mosaic.ipred'
        if key in intiles.columns:
            return intiles[key]
        arrays = self.xtile, self.ytile
        loc = pd.MultiIndex.from_arrays(arrays)
        result = (
            intiles.predtiles.ipred
            .loc[loc]
            .values
        )
        intiles[key] = result
        return intiles[key]

    @property
    def file(self) -> pd.Series:
        """predtiles.file broadcasted to intiles"""
        intiles = self.intiles
        key = 'mosaic.file'
        if key in intiles.columns:
            return intiles[key]
        result = (
            intiles.predtiles.file
            .loc[self.index]
            .values
        )
        intiles[key] = result
        return intiles[key]

    @property
    def skip(self) -> pd.Series:
        """predtiles.skip broadcasted to intiles"""
        intiles = self.intiles
        key = 'mosaic.skip'
        if key in intiles.columns:
            return intiles[key]
        result = (
            intiles.predtiles.skip
            .loc[self.index]
            .values
        )
        intiles[key] = result
        return intiles[key]


    def __init__(
            self,
            *args,
            **kwargs
    ):
        ...
