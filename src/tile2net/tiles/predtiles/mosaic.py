from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd
from .. import tile

if False:
    from .predtiles import PredTiles


def __get__(
        self: Mosaic,
        instance: PredTiles,
        owner: type[PredTiles],
) -> Mosaic:
    self.predtiles = instance
    self.PredTiles = owner
    return self


class Mosaic(

):
    predtiles: PredTiles
    locals().update(
        __get__=__get__,
    )

    @tile.cached_property
    def length(self):
        """Number of tiles in one dimension of the mosaic"""
        length = int(
            self.predtiles.outtiles.tile.scale
            - self.predtiles.tile.scale
        )
        return length ** 2


    @tile.cached_property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the outtiles mosaic"""
        key = 'mosaic.xtile'
        predtiles = self.predtiles
        if key in predtiles.columns:
            return predtiles[key]

        outtiles = predtiles.outtiles
        result = predtiles.xtile // predtiles.mosaic.length

        msg = 'All mosaic.xtile must be in outtiles.xtile!'
        assert result.isin(outtiles.xtile).all(), msg
        predtiles[key] = result
        return result

    @tile.cached_property
    def ytile(self) -> pd.Series:
        """Tile integer X of this tile in the outtiles mosaic"""
        predtiles = self.predtiles
        outtiles = predtiles.outtiles
        result = predtiles.ytile // predtiles.mosaic.length

        # todo: for each outtile origin, use array ranges to pad the tiles

    @tile.cached_property
    def frame(self) -> pd.DataFrame:
        predtiles = self.predtiles
        concat = []
        ytile = predtiles.ytile // predtiles.mosaic.length
        xtile = predtiles.xtile // predtiles.mosaic.length
        frame = pd.DataFrame(dict(
            xtile=xtile,
            ytile=ytile,
        ), index=predtiles.index)
        concat.append(frame)


    @property
    def ytile(self):
        return self.frame.ytile

    @property
    def xtile(self):
        return self.frame.xtile

    @property
    def group(self) -> pd.Series:
        predtiles = self.predtiles
        key = 'mosaic.group'
        if key in predtiles.columns:
            return predtiles[key]
        arrays = self.xtile, self.ytile
        loc = pd.MultiIndex.from_arrays(arrays)
        result = (
            predtiles.outtiles.ipred
            .loc[loc]
            .values
        )
        predtiles[key] = result
        return predtiles[key]

    def __init__(
            self,
            *args,
            **kwargs
    ):
        ...
