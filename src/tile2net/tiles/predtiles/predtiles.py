from __future__ import annotations
from .stitch import Stitch
from .predict import Predict

from typing import *

import numpy as np
import pandas as pd
import rasterio

from .batchiterator import BatchIterator
from .mosaic import Mosaic
from .. import tile
from ..tiles import Tiles
from ..outtiles import OutTiles

if False:
    from ..intiles import InTiles


class Tile(
    tile.Tile
):
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
        instance: Tiles,
        owner: type[Tiles],
) -> PredTiles:
    if instance is None:
        return self
    try:
        result = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'InTiles must be predtiles using `InTiles.stitch` for '
            f'example `InTiles.stitch.to_resolution(2048)` or '
            f'`InTiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.tiles = instance
    return result


class PredTiles(
    Tiles,
):
    __name__ = 'predtiles'

    @tile.cached_property
    def intiles(self) -> InTiles:
        ...

    @OutTiles
    def outtiles(self):
        """
        After performing PredTiles.stitch, PredTiles.outtiles is
        available for performing inference on the stitched tiles.
        """

    @Predict
    def predict(self):
        # This code block is just semantic sugar and does not run.
        # Take a look at the following methods which do run:
        result = (
            self.predict
            .with_polygons(
                max_hole_area=dict(
                    road=30,
                    crosswalk=15,
                ),
                grid_size=.001,
            )
            .to_outdir()
        )
        result = self.predict.to_outdir()

    @Stitch
    def stitch(self):
        # This code block is just semantic sugar and does not run.
        # Take a look at the following methods which do run:

        # stitch to a target resolution e.g. 2048 ptxels
        self.stitch.to_dimension(...)
        # stitch to a cluster size e.g. 16 tiles
        self.stitch.to_mosaic(...)
        # stitch to an XYZ scale e.g. 17
        self.stitch.to_scale(...)

    @tile.cached_property
    def intiles(self) -> InTiles:
        """InTiles object that this PredTiles object is based on"""

    @property
    def ipred(self) -> pd.Series:
        key = 'ipred'
        if key in self.columns:
            return self[key]
        self[key] = np.arange(len(self), dtype=np.uint32)
        return self[key]

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
        )

    @Tile
    def tile(self):
        ...
