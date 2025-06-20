from __future__ import annotations

from .. import inftiles
import pandas as pd

from ..tiles import Tiles, tile


class GeometryTile(
    inftiles.GeometryTile
):
    @property
    def r(self) -> pd.Series:
        """row within the mosaic of this tile"""
        inftiles = self.inftiles
        key = 'mosaic.r'
        if key in inftiles.columns:
            return inftiles[key]
        raise AttributeError

    @property
    def c(self) -> pd.Series:
        """column within the mosaic of this tile"""
        inftiles = self.inftiles
        key = 'mosaic.c'
        if key in inftiles.columns:
            return inftiles[key]
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
        """geotiles.file broadcasted to inftiles"""
        inftiles = self.inftiles
        key = 'mosaic.file'
        if key in inftiles.columns:
            return inftiles[key]
        result = (
            inftiles.geotiles.file
            .loc[self.index]
            .values
        )
        inftiles[key] = result
        return inftiles[key]

    @property
    def skip(self) -> pd.Series:
        """geotiles.skip broadcasted to inftiles"""
        inftiles = self.inftiles
        key = 'mosaic.skip'
        if key in inftiles.columns:
            return inftiles[key]
        result = (
            inftiles.geotiles.skip
            .loc[self.index]
            .values
        )
        inftiles[key] = result
        return inftiles[key]

