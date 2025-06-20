from __future__ import annotations

import pandas as pd
from ..tiles import tile

if False:
    from .inftiles import InferenceTiles


def __get__(
        self: GeometryTile,
        instance: InferenceTiles,
        owner: type[InferenceTiles],
) -> GeometryTile:
    self.inftiles = instance
    self.InferenceTiles = owner
    return self


class GeometryTile(

):
    inftiles: InferenceTiles
    locals().update(
        __get__=__get__,
    )

    @tile.cached_property
    def length(self):
        """Number of tiles in one dimension of the mosaic"""
        length = int(
            self.inftiles.geotiles.tile.scale
            - self.inftiles.tile.scale
        )
        return length ** 2


    @tile.cached_property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the geotiles mosaic"""
        key = 'mosaic.xtile'
        inftiles = self.inftiles
        if key in inftiles.columns:
            return inftiles[key]

        geotiles = inftiles.geotiles
        result = inftiles.xtile // inftiles.outtile.length

        msg = 'All mosaic.xtile must be in geotiles.xtile!'
        assert result.isin(geotiles.xtile).all(), msg
        inftiles[key] = result
        return result

    @tile.cached_property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the geotiles mosaic"""
        inftiles = self.inftiles
        geotiles = inftiles.geotiles
        result = inftiles.ytile // inftiles.outtile.length

        # todo: for each outtile origin, use array ranges to pad the tiles

    @tile.cached_property
    def frame(self) -> pd.DataFrame:
        inftiles = self.inftiles
        concat = []
        ytile = inftiles.ytile // inftiles.outtile.length
        xtile = inftiles.xtile // inftiles.outtile.length
        frame = pd.DataFrame(dict(
            xtile=xtile,
            ytile=ytile,
        ), index=inftiles.index)
        concat.append(frame)


    @property
    def ytile(self):
        return self.frame.ytile

    @property
    def xtile(self):
        return self.frame.xtile

    @property
    def group(self) -> pd.Series:
        inftiles = self.inftiles
        key = 'mosaic.group'
        if key in inftiles.columns:
            return inftiles[key]
        arrays = self.xtile, self.ytile
        loc = pd.MultiIndex.from_arrays(arrays)
        result = (
            inftiles.geotiles.ipred
            .loc[loc]
            .values
        )
        inftiles[key] = result
        return inftiles[key]


    def __init__(
            self,
            *args,
            **kwargs
    ):
        ...
