from __future__ import annotations
from tqdm import tqdm

import os.path

from pathlib import Path
from sympy.core.benchmarks.bench_assumptions import timeit_x_is_integer
from tempfile import gettempdir
from uuid import uuid4

import tempfile

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import imageio.v2

import math
import numpy as np
import pandas as pd
from typing import *
from .indir import Loader
from tile2net.logger import logger

if False:
    from .tiles import Tiles


class Stitch:
    tiles: Tiles

    def to_dimension(
            self,
            dimension: int = 1024,
            pad: bool = True,
    ) -> Tiles:
        """
        resolution:
            target resolution of the stitched tiles
        pad:
            True:
                Go back and include more tiles to ensure the stitched
                tiles include all the original tiles.
            False:
                Drop any tiles that cannot be stitched into a complete
                tile.

        Returns:
            Tiles:
                New Tiles object with Tiles.stitched set to the stitched
                tiles at the specified resolution.
        """
        tiles = self.tiles
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
        result = tiles.stitch.to_scale(scale, pad=pad)
        return result

    def to_mosaic(
            self,
            mosaic: int = 16,
            pad: bool = True,
    ) -> Tiles:
        """
        mosaic:
            Mosaic size of the stitched tiles. Must be a power of 2.
        pad:
            True:
                Go back and include more tiles to ensure the stitched
                tiles include all the original tiles.
            False:
                Drop any tiles that cannot be stitched into a complete
                tile.

        Returns:
            Tiles:
                New Tiles object with Tiles.stitched set to the stitched
                tiles at the specified cluster size.
        """
        if (
                not isinstance(mosaic, int)
                or mosaic <= 0
                or (mosaic & (mosaic - 1)) != 0
        ):
            raise ValueError('Cluster must be a positive power of 2.')
        tiles = self.tiles
        marea = int(math.log2(mosaic))
        dscale = int(math.sqrt(marea))
        scale = tiles.tscale - dscale
        result = tiles.stitch.to_scale(scale, pad=pad)
        return result

    def to_scale(
            self,
            scale: int = 17,
            pad: bool = True,
    ) -> Tiles:
        """
        scale:
            Scale of larger slippy tiles into which smaller tiles are stitched
        pad:
            True:
                Go back and include more tiles to ensure the stitched
                tiles include all the original tiles.
            False:
                Drop any tiles that cannot be stitched into a complete
                tile.
        """
        from .tiles import Tiles
        TILES = self.tiles

        tiles = TILES

        dscale = tiles.tscale - scale
        # todo: how to get the bounds in a different scale
        if dscale < 0:
            msg = (
                f'Cannot stitch from slippy scale {tiles.tscale} to '
                f'slippy scale {scale}. The target must be a lower int.'
            )
            raise ValueError(msg)
        mlength = dscale ** 2  # mosaic length
        marea = dscale ** 4  # mosaic area

        txy = tiles.index.to_frame()
        if txy.duplicated().any():
            msg = ('Cannot stitch tiles with duplicate indices!')
            raise ValueError(msg)

        # a mosaic is a group of tiles
        if pad:
            # fill all implicated mosaics
            mxy: pd.DataFrame = txy // mlength
            topleft = (
                mxy
                .drop_duplicates()
                .mul(mlength)
            )

            arange = np.arange(mlength)
            x, y = np.meshgrid(arange, arange, indexing='ij')
            txy: np.ndarray = (
                np.stack((x, y), -1)
                .reshape(-1, 2)
                .__add__(topleft.values[:, None, :])
                .reshape(-1, 2)
            )
            tx = txy[:, 0]
            ty = txy[:, 1]
            assert not pd.DataFrame(dict(tx=tx, ty=ty)).duplicated().any()
            tiles = Tiles.from_integers(tx, ty, zoom=tiles.zoom)
            tiles.attrs.update(TILES.attrs)
        else:
            # drop mosaics not containing all tiles
            mxy: pd.DataFrame = txy // mlength
            loc = (
                pd.Series(1, index=mxy)
                .groupby(mxy)
                .sum()
                .eq(mlength * mlength)
                .loc[mxy]
            )
            tiles = tiles.loc[loc]

        mxy = (
            tiles.index
            .to_frame()
            .floordiv(mlength)
        )
        unique = mxy.drop_duplicates()
        tx = unique.xtile
        ty = unique.ytile
        from .stitched import Stitched
        # tiles.indir.root +
        indir = tiles.indir
        indir = os.path.join(
            indir.dir,
            'stitched',
            indir.suffix,
        )
        _stitched = Tiles.from_integers(tx=tx, ty=ty, zoom=scale)
        stitched = Stitched(_stitched)
        stitched.attrs.update(_stitched.attrs)
        stitched.indir = indir
        tiles.stitched = stitched

        stitched.zoom = tiles.zoom
        stitched.tscale = scale
        mlength = dscale ** 2

        msg = f'Tile count is not {marea}x the mosaic count.'
        assert len(tiles) == len(tiles.stitched) * marea, msg
        msg = f'Not all mosaics are complete'
        assert (
            tiles
            .groupby(pd.MultiIndex.from_frame(mxy))
            .size()
            .eq(marea)
            .all()
        ), msg

        tiles: Tiles

        try:
            indir = tiles.indir
        except (AttributeError, KeyError) as e:
            ...
        else:

            # todo: do not stitch if file exists
            msg = (
                f'Stitching tiles from {tiles.indir.original} '
                f'to {stitched.indir.original}. '
            )
            logger.info(msg)
            # m = tiles.mosaic
            # iloc = np.lexsort((m.c, m.r, m.xtile, m.ytile))
            # infiles = tiles.file.iloc[iloc]
            # arrays = m.ytile.iloc[iloc], m.xtile.iloc[iloc]
            #
            # needles = pd.MultiIndex.from_arrays(arrays)
            # infiles = infiles.set_axis(infiles)
            # cols = 'mosaic.xtile mosaic.ytile'.split()
            # loc = (
            #     tiles[cols]
            #     .pipe(pd.MultiIndex.from_frame)
            #     .unique()
            # )
            # outfiles = stitched.file.loc[loc]
            # loc = [
            #     not os.path.exists(outfile)
            #     for outfile in outfiles
            # ]
            # outfiles = outfiles.loc[loc]
            # haystack = outfiles.index
            # loc = needles.isin(haystack)
            # infiles = infiles.loc[loc]
            #
            # total = len(tiles)
            # missing = len(infiles)
            # msg = (
            #     f'Found {total} tiles, stitching {missing} tiles '
            #     f'into {len(outfiles)} stitched tiles. '
            # )
            # logger.info(msg)
            # tiles.file
            # tiles.indir
            # infiles
            #
            # shape = (stitched.dimension, stitched.dimension, 3)
            # loader = Loader(infiles, shape)

            # iloc = np.argsort(stitched.group.values)
            # stitched.tiles
            # stitched = stitched.iloc[iloc]
            #
            # outfiles = stitched.file
            # tile = (tiles.dimension, tiles.dimension, 3)
            # mosaic = (stitched.dimension, stitched.dimension, 3)
            # loader = Loader(
            #     files=tiles.file,
            #     row=tiles.mosaic.r,
            #     col=tiles.mosaic.c,
            #     group=tiles.mosaic.group,
            #     tile=tile,
            #     mosaic=mosaic,
            # )
            #
            # executor = ThreadPoolExecutor()
            # imwrite = imageio.v3.imwrite
            # it = tqdm(
            #     zip(loader, outfiles),
            #     total=len(outfiles),
            #     desc='loading',
            #     unit='img'
            # )
            #
            # writes = []
            # for array, outfile in it:
            #     Path(outfile).parent.mkdir(parents=True, exist_ok=True)
            #     writes.append(executor.submit(imwrite, outfile, array))
            #
            # for w in writes:
            #     w.result()

            # sort stitched by group so that outfiles come out in row–major mosaic order
            iloc = np.argsort(stitched.group.values)
            stitched = stitched.iloc[iloc]
            outfiles = stitched.file  # 1:1 with mosaics

            # ── determine which mosaics are still missing ────────────────────────────────
            missing_mask = ~outfiles.apply(os.path.exists)
            n_total = len(outfiles)
            n_missing = int(missing_mask.sum())

            if n_missing == 0:  # nothing to do
                return tiles
            logger.info(f'Stitching {n_missing:,} of {n_total:,} mosaics (missing on disk).')

            # groups (integer IDs) whose stitched files are absent
            groups_needed = stitched.group[missing_mask].unique()
            tiles.mosaic.r
            tiles.mosaic.c

            # subset all per-tile Series to only tiles belonging to those groups
            sel = tiles.mosaic.group.isin(groups_needed).values
            files_sub = tiles.file.loc[sel]
            row_sub = tiles.mosaic.r.loc[sel]
            col_sub = tiles.mosaic.c.loc[sel]
            group_sub = tiles.mosaic.group.loc[sel]
            tiles

            tile_shape = (tiles.dimension, tiles.dimension, 3)
            mosaic_shape = (stitched.dimension, stitched.dimension, 3)

            loader = Loader(
                files=files_sub,
                row=row_sub,
                col=col_sub,
                group=group_sub,
                tile=tile_shape,
                mosaic=mosaic_shape,
            )

            # ── write the (still-missing) stitched mosaics in parallel ───────────────────
            executor = ThreadPoolExecutor()
            imwrite = imageio.v3.imwrite

            it = tqdm(
                zip(loader, outfiles[missing_mask]),
                total=n_missing,
                desc='stitching',
                unit='mosaic'
            )

            writes = []
            for array, outfile in it:
                Path(outfile).parent.mkdir(parents=True, exist_ok=True)
                writes.append(executor.submit(imwrite, outfile, array))

            for w in writes:
                w.result()

            executor.shutdown(wait=True)
            # executor.shutdown(wait=True)

        return tiles

    def __get__(
            self,
            instance,
            owner
    ) -> Self:
        self.tiles = instance
        self.Tiles = owner
        return self

    def __init__(self, *args, **kwargs):
        ...
