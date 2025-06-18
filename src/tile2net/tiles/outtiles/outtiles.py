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
from . import delayed



def __get__(
        self: OutTiles,
        instance: Optional[PredTiles],
        owner: type[Tiles],
) -> OutTiles:
    if instance is None:
        return self
    try:
        result: OutTiles = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'OutTiles must be stitched using `PredTiles.stitch` for '
            f'example `PredTiles.stitch.to_dimension(2048)` or '
            f'`PredTiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.predtiles = instance
    return result


class OutTiles(
    Tiles
):
    __name__ = 'outtiles'

    @delayed.Padded
    def padded(self) -> Padded:
        ...

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
