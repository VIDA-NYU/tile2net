from __future__ import annotations

from .. import segtiles
import pandas as pd

from ..tiles import Tiles, tile


class GeoTile(
    segtiles.GeoTile
):
    @property
    def r(self) -> pd.Series:
        """row within the mosaic of this tile"""
        segtiles = self.segtiles
        key = 'mosaic.r'
        if key in segtiles.columns:
            return segtiles[key]
        raise AttributeError

    @property
    def c(self) -> pd.Series:
        """column within the mosaic of this tile"""
        segtiles = self.segtiles
        key = 'mosaic.c'
        if key in segtiles.columns:
            return segtiles[key]
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
        """geotiles.file broadcasted to segtiles"""
        segtiles = self.segtiles
        key = 'mosaic.file'
        if key in segtiles.columns:
            return segtiles[key]
        result = (
            segtiles.geotiles.file
            .loc[self.index]
            .values
        )
        segtiles[key] = result
        return segtiles[key]

    @property
    def skip(self) -> pd.Series:
        """geotiles.skip broadcasted to segtiles"""
        segtiles = self.segtiles
        key = 'mosaic.skip'
        if key in segtiles.columns:
            return segtiles[key]
        result = (
            segtiles.geotiles.skip
            .loc[self.index]
            .values
        )
        segtiles[key] = result
        return segtiles[key]

