from __future__ import annotations

import os.path
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio.v2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from tile2net.tiles.logger import logger
from tile2net.tiles.dir.loader import Loader

if False:
    from .intiles import InTiles

def __get__(
        self: Stitch,
        instance: InTiles,
        owner
) -> Stitch:
    self.intiles = instance
    return self



class Stitch:
    intiles: InTiles

    def to_dimension(
            self,
            dimension: int = 1024,
            fill: bool = True,
    ) -> InTiles:
        """
        resolution:
            target resolution of the stitched tiles
        fill:
            True:
                Go back and include more tiles to ensure the stitched
                tiles include all the original tiles.
            False:
                Drop any tiles that cannot be stitched into a complete
                tile.

        Returns:
            InTiles:
                New InTiles object with InTiles.stitched set to the stitched
                tiles at the specified resolution.
        """
        tiles = self.intiles
        try:
            _ = tiles.dimension
        except KeyError as e:
            msg = (
                'You cannot stitch tiles to a resolution when the tile '
                'resolution has not yet been set. First you must perform'
                ' tiles.with_indir() or tiles.with_source() for the '
                'resolution to be determined.'
            )
            raise ValueError(msg) from e
        dscale = int(math.log2(dimension / tiles.dimension))
        scale = tiles.tscale - dscale
        result = tiles.stitch.to_scale(scale, fill=fill)
        return result


    def to_mosaic(
            self,
            mosaic: int = 16,
            fill: bool = True,
    ) -> InTiles:
        """
        mosaic:
            Mosaic size of the stitched tiles. Must be a power of 2.
        fill:
            True:
                Go back and include more tiles to ensure the stitched
                tiles include all the original tiles.
            False:
                Drop any tiles that cannot be stitched into a complete
                tile.

        Returns:
            InTiles:
                New InTiles object with InTiles.stitched set to the stitched
                tiles at the specified cluster size.
        """
        if (
                not isinstance(mosaic, int)
                or mosaic <= 0
                or (mosaic & (mosaic - 1)) != 0
        ):
            raise ValueError('Cluster must be a positive power of 2.')
        tiles = self.intiles
        marea = int(math.log2(mosaic))
        dscale = int(math.sqrt(marea))
        scale = tiles.tscale - dscale
        result = tiles.stitch.to_scale(scale, fill=fill)
        return result

    def to_scale(
            self,
            scale: int,
            fill: bool = True,
    ) -> InTiles:
        """
        scale:
            Scale of larger slippy tiles into which smaller tiles are stitched
        fill:
            True:
                Go back and include more tiles to ensure the stitched
                tiles include all the original tiles.
            False:
                Drop any tiles that cannot be stitched into a complete
                tile.
        """
        msg = 'Padding InTiles to align with PredTiles'
        logger.debug(msg)
        intiles = (
            self.intiles
            .to_scale(scale)
            .to_scale(self.intiles.tile.scale)
            .download()
        )
        predtiles = (
            intiles
            .to_scale(scale)
            .pipe(self.intiles.__class__.predtiles.__class__)
        )
        assert intiles.predtile.ipred.is_monotonic_increasing
        assert predtiles.ipred.is_monotonic_increasing
        intiles.predtiles = predtiles
        msg = 'Done padding.'
        logger.debug(msg)

        loc = ~intiles.predtile.skip
        infiles = intiles.file.loc[loc]
        row = intiles.predtile.r.loc[loc]
        col = intiles.predtile.c.loc[loc]
        group = intiles.predtile.ipred.loc[loc]

        loc = ~predtiles.skip
        predfiles = predtiles.file.loc[loc]
        n_missing = np.sum(loc)
        n_total = len(predtiles)

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
            mosaic_shape=predtiles.tile.shape,
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

        del predtiles.skip
        assert predtiles.skip.all()

        return intiles

    def __init__(self, *args, **kwargs):
        ...
