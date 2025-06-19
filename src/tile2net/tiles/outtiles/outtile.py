from __future__ import annotations

from .. import predtiles
import pandas as pd

from ..tiles import Tiles, tile


class OutTile(
    predtiles.OutTile
):
    @property
    def r(self) -> pd.Series:
        """row within the mosaic of this tile"""
        predtiles = self.predtiles
        key = 'mosaic.r'
        if key in predtiles.columns:
            return predtiles[key]
        raise AttributeError

    @property
    def c(self) -> pd.Series:
        """column within the mosaic of this tile"""
        predtiles = self.predtiles
        key = 'mosaic.c'
        if key in predtiles.columns:
            return predtiles[key]
        raise AttributeError

    @tile.cached_property
    def index(self):
        xtile = self.xtile
        ytile = self.ytile
        r = self.r
        c = self.c
        arrays = xtile, ytile, r, c
        names = xtile.name, ytile.name, r.name, c.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @property
    def file(self) -> pd.Series:
        """outtiles.file broadcasted to predtiles"""
        predtiles = self.predtiles
        key = 'mosaic.file'
        if key in predtiles.columns:
            return predtiles[key]
        result = (
            predtiles.outtiles.file
            .loc[self.index]
            .values
        )
        predtiles[key] = result
        return predtiles[key]

    @property
    def skip(self) -> pd.Series:
        """outtiles.skip broadcasted to predtiles"""
        predtiles = self.predtiles
        key = 'mosaic.skip'
        if key in predtiles.columns:
            return predtiles[key]
        result = (
            predtiles.outtiles.skip
            .loc[self.index]
            .values
        )
        predtiles[key] = result
        return predtiles[key]

