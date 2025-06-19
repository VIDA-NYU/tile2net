from __future__ import annotations

from ..dir import BatchIterator
from typing import *

import numpy as np
import pandas as pd
import rasterio

from ..tiles import Tiles, tile


if False:
    from .padded import Padded
    from ..predtiles import PredTiles

from ..tiles import Tiles
from . import delayed

def __get__(
        self: Padding,
        instance: OutTiles,
        owner,
) -> Padding:
    self.outtiles = instance
    return self


class Padding(

):
    outtiles: OutTiles = None
    locals().update(
        __get__=__get__,
    )

    @property
    def gw(self) -> pd.Series:
        outtiles = self.outtiles
        padded = outtiles.padded
        haystack = padded.outtile.index
        index = outtiles.index
        top_left = np.zeros_like((len(index), 2))
        needles = pd.MultiIndex.append(index, top_left)
        result = (
            padded
            .set_axis(haystack)
            .loc[needles, 'gw']
            .values
        )
        raise NotImplementedError

    @property
    def gn(self) -> pd.Series:
        ...

    @property
    def ge(self) -> pd.Series:
        ...

    @property
    def gs(self) -> pd.Series:
        ...



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
    def affine_params(self) -> pd.Series:
        key = 'affine_params'
        if key in self:
            return self[key]

        dim = self.tile.dimension
        self: pd.DataFrame
        col = 'gw gs ge gn'.split()
        it = self[col].itertuples(index=False)
        data = [
            rasterio.transform
            .from_bounds(gw, gs, ge, gn, dim, dim)
            for gw, gs, ge, gn in it
        ]
        result = pd.Series(data, index=self.index, name=key)
        self[key] = result
        return self[key]

    @BatchIterator
    def affine_iterator(self):
        return self.affine_params

    @property
    def skip(self):
        key = 'skip'
        if key in self:
            return  self[key]
        self[key] = self.intiles.outdir.outtiles.skip()
        return self[key]

    @property
    def file(self):
        key = 'file'
        if key in self:
            return self[key]
        self[key] = self.intiles.outdir.outtiles.files()
        return self[key]
