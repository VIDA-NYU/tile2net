from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import *

import imageio.v2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tile2net.tiles.dir.loader import Loader
from tile2net.tiles.cfg.logger import logger
from .predict import Predict
from .vectile import VecTile
from ..tiles import tile
from ..tiles.tiles import Tiles

if False:
    from ..intiles import InTiles


class Tile(
    tile.Tile
):
    tiles: SegTiles

    @tile.cached_property
    def length(self) -> int:
        """How many input tiles comprise a segmentation tile"""
        intiles = self.tiles.intiles
        result = 2 ** (intiles.tile.scale - self.scale)
        return result

    @tile.cached_property
    def dimension(self):
        """How many pixels in a inmentation tile"""
        segtiles = self.tiles
        intiles = segtiles.intiles
        result = intiles.tile.dimension * self.length
        return result


def __get__(
        self: SegTiles,
        instance: Tiles,
        owner: type[Tiles],
) -> SegTiles:
    if instance is None:
        return self
    try:
        result = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'InTiles must be segtiles using `InTiles.stitch` for '
            f'example `InTiles.stitch.to_resolution(2048)` or '
            f'`InTiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.intiles = instance
    return result


class SegTiles(
    Tiles,
):
    __name__ = 'segtiles'
    locals().update(
        __get__=__get__,
    )

    @tile.cached_property
    def intiles(self) -> InTiles:
        ...

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

    def stitch(
            self
    ):
        intiles = self.intiles
        segtiles = self

        loc = ~intiles.segtile.skip
        infiles = intiles.infile.loc[loc]
        row = intiles.segtile.r.loc[loc]
        col = intiles.segtile.c.loc[loc]
        group = intiles.segtile.ipred.loc[loc]

        loc = ~segtiles.skip
        predfiles = segtiles.infile.loc[loc]
        n_missing = np.sum(loc)
        n_total = len(segtiles)

        if n_missing == 0:  # nothing to do
            msg = f'All {n_total:,} mosaics are already stitched.'
            logger.info(msg)
            return intiles
        else:
            logger.info(f'Stitching {n_missing:,} of {n_total:,} mosaics missing on disk.')

        loader = Loader(
            files=infiles,
            row=row,
            col=col,
            tile_shape=intiles.tile.shape,
            mosaic_shape=segtiles.tile.shape,
            group=group
        )

        seen = set()
        for f in predfiles:
            d = Path(f).parent
            if d not in seen:  # avoids extra mkdir syscalls
                d.mkdir(parents=True, exist_ok=True)
                seen.add(d)

        executor = ThreadPoolExecutor()
        imwrite = imageio.v3.imwrite
        it = zip(loader, predfiles)
        it = tqdm(it, 'stitching', n_missing, unit=' mosaic')

        writes = [
            executor.submit(imwrite, outfile, array)
            for array, outfile in it
        ]

        for w in writes:
            w.result()

        executor.shutdown(wait=True)

        del segtiles.skip
        assert segtiles.skip.all()

        return intiles

    @tile.cached_property
    def intiles(self) -> InTiles:
        """InTiles object that this SegTiles object is based on"""

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

    @VecTile
    def vectile(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the tiles have been stitched:
        # xtile of the vectile
        self.vectiles.xtile = ...
        # ytile of vectile
        self.vectiles.ytile = ...

    @Tile
    def tile(self):
        ...

    @property
    def infile(self) -> pd.Series:
        """
        A file for each segmentation tile: the stitched input tiles.
        Stitches input files when segtiles.file is accessed
        """
        key = 'infile'
        if key in self:
            return self[key]
        self[key] = self.intiles.outdir.segtiles.files()
        if not self.skip.all():
            self.stitch()
        return self[key]

    @property
    def skip(self) -> pd.Series:
        key = 'skip'
        if key in self:
            return self[key]
        self[key] = self.intiles.outdir.segtiles.skip()
        return self[key]

    @property
    def segtiles(self) -> Self:
        return self
