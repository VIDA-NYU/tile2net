from __future__ import annotations
from tqdm import tqdm

import os.path

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import imageio.v2

import math
import numpy as np
import pandas as pd
from tile2net.grid.dir.loader import Loader
from tile2net.grid.cfg.logger import logger

if False:
    from .ingrid import InGrid

def __get__(
        self: Stitch,
        instance: InGrid,
        owner
) -> Stitch:
    self.ingrid = instance
    return self



class Stitch:
    ingrid: InGrid

    def to_dimension(
            self,
            dimension: int = 1024,
            pad: bool = True,
    ) -> InGrid:
        """
        resolution:
            target resolution of the stitched grid
        pad:
            True:
                Go back and include more grid to ensure the stitched
                grid include all the original grid.
            False:
                Drop any grid that cannot be stitched into a complete
                tile.

        Returns:
            InGrid:
                New InGrid object with InGrid.stitched set to the stitched
                grid at the specified resolution.
        """
        grid = self.ingrid
        try:
            _ = grid.dimension
        except KeyError as e:
            msg = (
                'You cannot stitch grid to a resolution when the tile '
                'resolution has not yet been set. First you must perform'
                ' grid.with_indir() or grid.with_source() for the '
                'resolution to be determined.'
            )
            raise ValueError(msg) from e
        dscale = int(math.log2(dimension / grid.dimension))
        scale = grid.tscale - dscale
        result = self.to_scale(scale, pad=pad)
        return result

    def to_mosaic(
            self,
            mosaic: int = 16,
            pad: bool = True,
    ) -> InGrid:
        """
        mosaic:
            Mosaic size of the stitched grid. Must be a power of 2.
        pad:
            True:
                Go back and include more grid to ensure the stitched
                grid include all the original grid.
            False:
                Drop any grid that cannot be stitched into a complete
                tile.

        Returns:
            InGrid:
                New InGrid object with InGrid.stitched set to the stitched
                grid at the specified cluster size.
        """
        if (
                not isinstance(mosaic, int)
                or mosaic <= 0
                or (mosaic & (mosaic - 1)) != 0
        ):
            raise ValueError('Cluster must be a positive power of 2.')
        grid = self.ingrid
        marea = int(math.log2(mosaic))
        dscale = int(math.sqrt(marea))
        scale = grid.tscale - dscale
        result = self.to_scale(scale, pad=pad)
        return result

    def to_scale(
            self,
            scale: int = 17,
            pad: bool = True,
    ) -> InGrid:
        """
        scale:
            Scale of larger slippy grid into which smaller grid are stitched
        pad:
            True:
                Go back and include more grid to ensure the stitched
                grid include all the original grid.
            False:
                Drop any grid that cannot be stitched into a complete
                tile.
        """
        from .ingrid import InGrid
        TILES = self.ingrid

        grid = TILES

        dscale = grid.tscale - scale
        # todo: how to get the bounds in a different scale
        if dscale < 0:
            msg = (
                f'Cannot stitch from slippy scale {grid.tscale} to '
                f'slippy scale {scale}. The target must be a lower int.'
            )
            raise ValueError(msg)
        mlength = dscale ** 2  # mosaic length
        marea = dscale ** 4  # mosaic area

        txy = grid.index.to_frame()
        if txy.duplicated().any():
            msg = ('Cannot stitch grid with duplicate indices!')
            raise ValueError(msg)

        # a mosaic is a group of grid
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
            grid = InGrid.from_integers(tx, ty, scale=grid.zoom)
            grid.__dict__.update(TILES.__dict__)
            if 'source' in TILES.__dict__:
                grid = grid.set_source(
                    source=TILES.source,
                    indir=TILES.indir.original,
                )
            # we need to download from source if appropriate
        else:
            # drop mosaics not containing all grid
            mxy: pd.DataFrame = txy // mlength
            loc = (
                pd.Series(1, index=mxy)
                .groupby(mxy)
                .sum()
                .eq(mlength * mlength)
                .loc[mxy]
            )
            grid = grid.loc[loc]

        mxy = (
            grid.index
            .to_frame()
            .floordiv(mlength)
        )
        unique = mxy.drop_duplicates()
        tx = unique.xtile
        ty = unique.ytile
        # grid.indir.root +
        indir = grid.indir
        indir = os.path.join(
            indir.dir,
            'stitched',
            indir.suffix,
        )
        _stitched = InGrid.from_integers(tx=tx, ty=ty, scale=scale)
        stitched = Stitched(_stitched)
        stitched.__dict__.update(_stitched.__dict__)
        setattr(stitched, 'indir', indir)
        grid.segtile = stitched
        stitched.grid = grid

        stitched.zoom = grid.zoom
        stitched.tscale = scale
        mlength = dscale ** 2

        msg = f'Tile count is not {marea}x the mosaic count.'
        assert len(grid) == len(grid.seggrid) * marea, msg
        msg = f'Not all mosaics are complete'
        assert (
            grid
            .groupby(pd.MultiIndex.from_frame(mxy))
            .size()
            .eq(marea)
            .all()
        ), msg

        grid: InGrid

        try:
            indir = grid.indir
        except (AttributeError, KeyError) as e:
            ...
        else:

            # todo: do not stitch if file exists
            msg = (
                f'Stitching grid from \n\t{grid.indir.original} '
                f'to\n\t{stitched.indir.original}. '
            )
            logger.info(msg)

            # sort stitched by group so that outfiles come out in row–major mosaic order
            iloc = np.argsort(stitched.ipred.values)
            stitched: Stitched = stitched.iloc[iloc]
            # outfiles = stitched.file  # 1:1 with mosaics
            outfiles = stitched.indir.files()

            # ── determine which mosaics are still missing ────────────────────────────────
            missing_mask = ~outfiles.apply(os.path.exists)
            n_total = len(outfiles)
            n_missing = int(missing_mask.sum())

            if n_missing == 0:  # nothing to do
                msg = f'All {n_total:,} mosaics are already stitched.'
                logger.info(msg)
                return grid
            else:
                logger.info(f'Stitching {n_missing:,} of {n_total:,} mosaics missing on disk.')

            # groups (integer IDs) whose stitched files are absent
            groups_needed = stitched.ipred[missing_mask].unique()

            # subset all per-tile Series to only grid belonging to those groups
            sel = grid.segtile.ipred.isin(groups_needed).values
            files_sub = grid.indir.files().loc[sel]

            row_sub = grid.segtile.r.loc[sel]
            col_sub = grid.segtile.c.loc[sel]
            group_sub = grid.segtile.ipred.loc[sel]

            tile_shape = grid.dimension, grid.dimension, 3
            mosaic_shape = stitched.dimension, stitched.dimension, 3

            loader = Loader(
                infiles=files_sub,
                row=row_sub,
                col=col_sub,
                outfiles=group_sub,
                tile_shape=tile_shape,
                mosaic_shape=mosaic_shape,
            )

            # ── write the (still-missing) stitched mosaics in parallel ───────────────────
            executor = ThreadPoolExecutor()
            imwrite = imageio.v3.imwrite

            it = zip(loader, outfiles[missing_mask])
            it = tqdm(it, 'stitching', n_missing, unit=' mosaic')

            writes = []
            for array, outfile in it:
                Path(outfile).parent.mkdir(parents=True, exist_ok=True)
                writes.append(executor.submit(imwrite, outfile, array))

            for w in writes:
                w.result()

            executor.shutdown(wait=True)

            files: pd.Series[str] = stitched.grayscale
            assert all(map(os.path.exists, files))

        return grid

    def __init__(self, *args, **kwargs):
        ...
