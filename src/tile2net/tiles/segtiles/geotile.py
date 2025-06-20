from __future__ import annotations

import copy

import pandas as pd
from ..tiles import tile

if False:
    from .segtiles import SegTiles


def __get__(
        self: GeoTile,
        instance: SegTiles,
        owner: type[SegTiles],
) -> GeoTile:
    self.segtiles = instance
    return copy.copy(self)


class GeoTile(

):
    segtiles: SegTiles
    locals().update(
        __get__=__get__,
    )
    @property
    def tiles(self):
        return self.segtiles

    @tile.cached_property
    def length(self):
        """Number of tiles in one dimension of the mosaic"""
        length = int(
            self.segtiles.geotiles.tile.scale
            - self.segtiles.tile.scale
        )
        return length ** 2


    @tile.cached_property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the geotiles mosaic"""
        key = 'mosaic.xtile'
        segtiles = self.segtiles
        if key in segtiles.columns:
            return segtiles[key]

        geotiles = segtiles.geotiles
        result = segtiles.xtile // segtiles.geotile.length

        msg = 'All mosaic.xtile must be in geotiles.xtile!'
        assert result.isin(geotiles.xtile).all(), msg
        segtiles[key] = result
        return result

    @tile.cached_property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the geotiles mosaic"""
        segtiles = self.segtiles
        geotiles = segtiles.geotiles
        result = segtiles.ytile // segtiles.geotile.length

        # todo: for each geotile origin, use array ranges to pad the tiles

    @tile.cached_property
    def frame(self) -> pd.DataFrame:
        segtiles = self.segtiles
        concat = []
        ytile = segtiles.ytile // segtiles.geotile.length
        xtile = segtiles.xtile // segtiles.geotile.length
        frame = pd.DataFrame(dict(
            xtile=xtile,
            ytile=ytile,
        ), index=segtiles.index)
        concat.append(frame)


    @property
    def ytile(self):
        return self.frame.ytile

    @property
    def xtile(self):
        return self.frame.xtile

    @property
    def group(self) -> pd.Series:
        segtiles = self.segtiles
        key = 'mosaic.group'
        if key in segtiles.columns:
            return segtiles[key]
        arrays = self.xtile, self.ytile
        loc = pd.MultiIndex.from_arrays(arrays)
        result = (
            segtiles.geotiles.ipred
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
