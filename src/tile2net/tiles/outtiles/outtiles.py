from __future__ import annotations

from ..dir import BatchIterator
from typing import *

import numpy as np
import pandas as pd
import rasterio

from ..tiles import Tiles, tile


if False:
    from .padded import Padded
    from ..inftiles import InferenceTiles

from ..tiles import Tiles
from . import delayed

def __get__(
        self: Padding,
        instance: GeometryTiles,
        owner,
) -> Padding:
    self.geotiles = instance
    return self


class Padding(

):
    geotiles: GeometryTiles = None
    locals().update(
        __get__=__get__,
    )

    @property
    def gw(self) -> pd.Series:
        geotiles = self.geotiles
        padded = geotiles.padded
        haystack = padded.outtile.index
        index = geotiles.index
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
        self: GeometryTiles,
        instance: Optional[InferenceTiles],
        owner: type[Tiles],
) -> GeometryTiles:
    if instance is None:
        return self
    try:
        result: GeometryTiles = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'GeometryTiles must be stitched using `InferenceTiles.stitch` for '
            f'example `InferenceTiles.stitch.to_dimension(2048)` or '
            f'`InferenceTiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.inftiles = instance
    return result

class GeometryTiles(
    Tiles
):
    __name__ = 'geotiles'

    @delayed.Padded
    def padded(self) -> Padded:
        ...

    @tile.cached_property
    def inftiles(self) -> InferenceTiles:
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
        self[key] = self.intiles.outdir.geotiles.skip()
        return self[key]

    @property
    def file(self):
        key = 'file'
        if key in self:
            return self[key]
        self[key] = self.intiles.outdir.geotiles.files()
        return self[key]
