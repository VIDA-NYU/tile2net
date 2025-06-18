from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd
import rasterio

from .. import tile
from ..tiles import Tiles

from .padded import Padded

if False:
    from ..intiles import InTiles
    from ..predtiles import PredTiles

from ..tiles import Tiles


def __get__(
        self: OutTiles,
        instance: Optional[PredTiles],
        owner: type[Tiles],
) -> OutTiles:
    if instance is None:
        return self
    try:
        result = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'Tiles must be predtiles using `Tiles.stitch` for '
            f'example `Tiles.stitch.to_resolution(2048)` or '
            f'`Tiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.intiles = instance
    return result


class OutTiles(
    Tiles
):
    __name__ = 'outtiles'

    @property
    def padded(self) -> Padded:
        from .padded import Padded
        setattr(self.__class__, 'padded', Padded())
        return self.padded


    @tile.cached_property
    def predtiles(self) -> PredTiles:
        ...

    @property
    def xorigin(self) -> pd.Series:
        key = 'xorigin'
        if key in self:
            return self[key]
        pred = self.predtiles
        result = pred.xtile // pred.mosaic.length
        result *= pred.mosaic.length
        self[key] = result
        return self[key]

    @property
    def yorigin(self) -> pd.Series:
        key = 'yorigin'
        if key in self:
            return self[key]
        pred = self.predtiles
        result = pred.ytile // pred.mosaic.length
        result *= pred.mosaic.length
        self[key] = result
        return self[key]
