from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
import os
from pathlib import Path
from typing import Tuple

import imageio.v3 as iio
import numpy as np

from concurrent.futures import Future
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import *

import imageio.v3 as iio
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm


def _read_one(path: str) -> np.ndarray | None:
    if Path(path).is_file():
        return iio.imread(path)
    return None


def _assemble_and_save(  # top-level → pickle-friendly
        files: list[str],
        rows: np.ndarray,
        cols: np.ndarray,
        out_path: str,
        tile_shape: Tuple[int, int, int],
        mosaic_shape: Tuple[int, int, int],
        dtype: np.dtype,
) -> str:
    """
    Worker routine executed in a separate process.  Reads *files*, places them
    into their correct slots, and writes the finished mosaic to *out_path*.
    Returns the destination path so the parent can assert success.
    """

    tile_h, tile_w, _ = tile_shape
    mos_h, mos_w, mos_c = mosaic_shape

    mosaic = np.zeros(mosaic_shape, dtype=dtype)

    for f, r, c in zip(files, rows, cols, strict=True):
        if not Path(f).is_file():
            # skip silently – upstream code decided the file might exist
            continue

        img = iio.imread(f)

        # harmonise channel count
        if img.ndim == 2:  # gray → colour
            img = np.repeat(img[..., None], mos_c, axis=2)
        elif img.shape[2] > mos_c:  # drop alpha, etc.
            img = img[..., :mos_c]
        elif img.shape[2] < mos_c:  # pad missing
            pad = np.zeros((
                *img.shape[:2],
                mos_c - img.shape[2],
            ), dtype=img.dtype)
            img = np.concatenate((img, pad), axis=2)

        y0 = r * tile_h
        x0 = c * tile_w
        mosaic[
        y0:y0 + tile_h,
        x0:x0 + tile_w,
        :
        ] = img

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, mosaic)

    return out_path


class Loader:
    """
    Process-based mosaic assembler.  One **process** per outfile.
    """

    def __init__(
            self,
            *,
            infiles: pd.Series,
            outfiles: pd.Series,
            row: pd.Series,
            col: pd.Series,
            tile_shape: Tuple[int, int, int],
            mosaic_shape: Tuple[int, int, int],
    ):

        if not (len(infiles) == len(outfiles) == len(row) == len(col)):
            raise ValueError(
                'infiles, outfiles, row and col must have the same length',
            )

        (
            self.files,
            self.group,
            self.row,
            self.col,
        ) = (
            s.reset_index(drop=True)
            for s in (infiles, outfiles, row, col)
        )

        self.tile_h, self.tile_w, self.tile_c = tile_shape
        self.mos_h, self.mos_w, self.mos_c = mosaic_shape

        if (self.mos_h % self.tile_h) or (self.mos_w % self.tile_w):
            raise ValueError('Mosaic dimensions must be multiples of tile size')

        # dtype discovery (first existing file – cheap)
        self.dtype = next(
            (
                iio.imread(p).dtype
                for p in self.files
                if Path(p).is_file()
            ),
            np.uint8,
        )

        # sort deterministically (group, row, col)
        order = np.lexsort((self.col, self.row, self.group))
        for attr in ('files', 'group', 'row', 'col'):
            setattr(self, attr, getattr(self, attr).iloc[order])

        self.unique_groups = self.group.unique()

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def run(
            self,
            *,
            max_workers: int | None = None,
    ) -> None:
        """
        Assemble and write *every* missing mosaic using a
        :class:`~concurrent.futures.ProcessPoolExecutor`.
        """

        with ProcessPoolExecutor(max_workers=max_workers) as ex:

            # ── 1️⃣  progress while we enqueue jobs ────────────────────────────
            enqueue_bar = tqdm(
                total=len(self.unique_groups),
                desc='stitching',
                unit=' segtile.file.infile',
                position=0,
                leave=False,
            )

            tasks: list[Future[str]] = []

            for g in self.unique_groups:
                enqueue_bar.update(1)  # update *before* the costly submit
                enqueue_bar.refresh()  # force a redraw right now

                mask = self.group == g
                fut = ex.submit(
                    _assemble_and_save,
                    self.files[mask].to_list(),
                    self.row[mask].to_numpy(),
                    self.col[mask].to_numpy(),
                    g,
                    (self.tile_h, self.tile_w, self.tile_c),
                    (self.mos_h, self.mos_w, self.mos_c),
                    self.dtype,
                )
                tasks.append(fut)

            enqueue_bar.close()

            # ── 2️⃣  progress while the processes finish ───────────────────────
            for fut in tqdm(
                    as_completed(tasks),
                    total=len(tasks),
                    desc='stitching',
                    unit=' mosaic',
                    position=1,
            ):
                fut.result()

    def run(
            self,
            *,
            max_workers: int | None = None,
    ) -> None:
        """
        Assemble and write every missing mosaic using
        :class:`concurrent.futures.ProcessPoolExecutor`, but build the work
        list via a single `DataFrame.groupby('outfile')` to avoid the
        per-iteration boolean mask.
        """
        print('⚠️AI GENERATED🤖')

        # ── build one tidy frame ------------------------------------------------
        df = (
            pd.DataFrame({
                'file': self.files,
                'outfile': self.group,  # one mosaic per unique *outfile*
                'row': self.row,
                'col': self.col,
            })
            .sort_values(['outfile', 'row', 'col'])
            .reset_index(drop=True)
        )

        groups = df.groupby('outfile', sort=False)

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            tasks: list[Future[str]] = []

            # ── enqueue all mosaics --------------------------------------------
            for out_path, g in tqdm(
                    groups,
                    total=groups.ngroups,
                    desc='queueing',
                    unit=' mosaic',
                    position=0,
                    leave=False,
            ):
                fut = ex.submit(
                    _assemble_and_save,
                    g.file.tolist(),
                    g.row.to_numpy(),
                    g.col.to_numpy(),
                    out_path,
                    (self.tile_h, self.tile_w, self.tile_c),
                    (self.mos_h, self.mos_w, self.mos_c),
                    self.dtype,
                )
                tasks.append(fut)

            # ── wait for completion -------------------------------------------
            for fut in tqdm(
                    as_completed(tasks),
                    total=len(tasks),
                    desc='stitching',
                    unit=' mosaic',
                    position=1,
            ):
                fut.result()  # propagate failures
