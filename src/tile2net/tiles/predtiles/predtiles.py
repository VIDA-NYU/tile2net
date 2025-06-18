from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd
import rasterio

from .batchiterator import BatchIterator
from .mosaic import Mosaic
from .. import tile
from ..tiles import Tiles

if False:
    from ..intiles import InTiles



class Tile(tile.Tile):
    tiles: PredTiles

    @property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        intiles = self.tiles.intiles
        key = 'tile.dimension'
        cache = intiles.attrs
        if key in cache:
            return cache[key]
        result = intiles.tile.dimension * intiles.mosaic.length
        cache[key] = result
        self.tiles.intiles.cfg.stitch.dimension = result
        return result

def __get__(
        self: PredTiles,
        instance: Optional[Tiles],
        owner: type[Tiles],
) -> PredTiles:
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


class PredTiles(
    Tiles,
):
    __name__ = 'predtiles'

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(
            self,
            instance: Tiles,
            value: type[Tiles],
    ):
        value.__name__ = self.__name__
        instance.attrs[self.__name__] = value



    @tile.cached_property
    def intiles(self) -> InTiles:
        """InTiles object that this PredTiles object is based on"""

    @property
    def group(self) -> pd.Series:
        if 'group' in self.columns:
            return self['group']
        result = pd.Series(np.arange(len(self)), index=self.index)
        self['group'] = result
        return self['group']

    @property
    def outdir(self):
        return self.intiles.outdir

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

    # def affine_iterator(self, *args, **kwargs) -> Iterator[ndarray]:
    #     key = 'affine_iterator'
    #     cache = self.intiles.attrs
    #     if key in cache:
    #         it = cache[key]
    #     else:
    #         affine = self.affine_params
    #         if not self.intiles.cfg.force:
    #             loc = ~self.intiles.outdir.skip
    #             affine = affine[loc]
    #
    #         def gen():
    #             n = self.cfg.model.bs_val
    #             a = affine.to_numpy()
    #             q, r = divmod(len(a), n)
    #             yield from a[:q * n].reshape(q, n)
    #             if r:
    #                 yield a[-r:]
    #
    #         it = gen()
    #         cache[key] = it
    #     yield from it
    #     del cache[key]

    @BatchIterator
    def affine_iterator(self):
        raise NotImplementedError('need to implement skip')
        return self.affine_params

    @property
    def cfg(self):
        return self.intiles.cfg

    @property
    def static(self):
        return self.intiles.static


    @Mosaic
    def mosaic(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the tiles have been stitched:
        _ = (
            # xtile of the larger mosaic
            self.mosaic.xtile,
            # ytile of the larger mosaic
            self.mosaic.ytile,
            # row of the tile within the larger mosaic
            self.mosaic.r,
            # column of the tile within the larger mosaic
            self.mosaic.c,
        )

    @Tile
    def tile(self):
        ...
