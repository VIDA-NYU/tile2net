from __future__ import annotations
from .stitch import Stitch
from .predict import Predict

import numpy as np
import pandas as pd

from .outtile import GeometryTile
from ..tiles import Tiles, tile
from ..geotiles import GeometryTiles

if False:
    from ..intiles import InTiles


class Tile(
    tile.Tile
):
    tiles: InferenceTiles

    @property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        intiles = self.tiles.intiles
        key = 'tile.dimension'
        cache = intiles.attrs
        if key in cache:
            return cache[key]
        result = intiles.tile.dimension * intiles.predtile.length
        cache[key] = result
        self.tiles.intiles.cfg.stitch.dimension = result
        return result


def __get__(
        self: InferenceTiles,
        instance: Tiles,
        owner: type[Tiles],
) -> InferenceTiles:
    if instance is None:
        return self
    try:
        result = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'InTiles must be inftiles using `InTiles.stitch` for '
            f'example `InTiles.stitch.to_resolution(2048)` or '
            f'`InTiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.intiles = instance
    return result


class InferenceTiles(
    Tiles,
):
    __name__ = 'inftiles'

    @tile.cached_property
    def intiles(self) -> InTiles:
        ...

    @GeometryTiles
    def geotiles(self):
        """
        After performing InferenceTiles.stitch, InferenceTiles.geotiles is
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
        """InTiles object that this InferenceTiles object is based on"""

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
    def cfg(self):
        return self.intiles.cfg

    @property
    def static(self):
        return self.intiles.static

    @GeometryTile
    def outtile(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the tiles have been stitched:
        # xtile of the outtile
        self.geotiles.xtile = ...
        # ytile of outtile
        self.geotiles.ytile = ...


    @Tile
    def tile(self):
        ...

    @property
    def file(self) -> pd.Series:
        key = 'file'
        if key in self:
            return self[key]
        self[key] = self.intiles.outdir.inftiles.files()
        return self[key]

    @property
    def skip(self) -> pd.Series:
        key = 'skip'
        if key in self:
            return self[key]
        self[key] = self.intiles.outdir.inftiles.skip()
        return self[key]
