from __future__ import annotations

from typing import *

import pandas as pd

if False:
    from .tiles import Tiles


def __get__(
        self: Mosaic,
        instance: Tiles,
        owner: type[Tiles],
) -> Self:
    self.tiles = instance
    self.Tiles = owner
    return self


class Mosaic(

):
    tiles: Tiles
    locals().update(
        __get__=__get__,
    )

    @property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the stitched mosaic"""
        tiles = self.tiles
        if 'mosaic.xtile' in tiles.columns:
            return tiles['mosaic.xtile']

        stitched = tiles.stitched
        dscale = tiles.tscale - stitched.tscale
        mdim = dscale ** 2  # mosaic length

        result = tiles.xtile // mdim
        msg = 'All mosaic.xtile must be in stitched.xtile!'
        assert result.isin(stitched.xtile).all(), msg
        tiles['mosaic.xtile'] = result
        return result

    @property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the stitched mosaic"""
        tiles = self.tiles
        if 'mosaic.ytile' in tiles.columns:
            return tiles['mosaic.ytile']

        stitched = tiles.stitched
        dscale = tiles.tscale - stitched.tscale
        mdim = dscale ** 2  # mosaic length

        result = tiles.ytile // mdim
        msg = 'All mosaic.ytile must be in stitched.ytile!'
        assert result.isin(stitched.ytile).all(), msg
        tiles['mosaic.ytile'] = result
        return result

    @property
    def r(self) -> pd.Series:
        """row within the mosaic of this tile"""
        tiles = self.tiles
        if 'mosaic.r' in tiles.columns:
            return tiles['mosaic.r']
        stitched = tiles.stitched
        result = (
            tiles.ytile
            .to_series(index=tiles.index)
            .floordiv(stitched.mlength)
            .mul(stitched.mlength)
            .rsub(tiles.ytile.values)
        )
        tiles['mosaic.r'] = result
        result = tiles['mosaic.r']
        return result

    @property
    def c(self) -> pd.Series:
        """column within the mosaic of this tile"""
        tiles = self.tiles
        if 'mosaic.c' in tiles.columns:
            return tiles['mosaic.c']
        stitched = tiles.stitched
        result = (
            tiles.xtile
            .to_series(index=tiles.index)
            .floordiv(stitched.mlength)
            .mul(stitched.mlength)
            .rsub(tiles.xtile.values)
        )
        tiles['mosaic.c'] = result
        result = tiles['mosaic.c']
        return result

    @property
    def group(self) -> pd.Series:
        tiles = self.tiles
        if 'mosaic.group' in tiles.columns:
            return tiles['mosaic.group']
        arrays = self.xtile, self.ytile
        loc = pd.MultiIndex.from_arrays(arrays)
        result = (
            tiles.stitched.group
            .loc[loc]
            .values
        )
        tiles['mosaic.group'] = result
        return tiles['mosaic.group']

    @property
    def px(self):
        """Starting pixel X coordinate of this tile in the stitched mosaic"""

    @property
    def py(self):
        """Starting pixel Y coordinate of this tile in the stitched mosaic"""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        ...
