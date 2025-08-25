from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import logging
import multiprocessing as mp
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
        outfile: str,
        tile_h: int,
        tile_w: int,
        mos_h: int,
        mos_w: int,
        dtype: np.dtype,
        background: int = 0,
) -> str:
    # allocate RGBA working canvas (α + RGB); write RGB only at the end
    mosaic = np.full((mos_h, mos_w, 4), background, dtype=dtype)
    mosaic[..., 3] = 255

    for f, r, c in zip(files, rows, cols, strict=True):
        # silently skip absent tiles
        if not Path(f).is_file():
            continue

        # always decode as RGBA for uniformity
        img = iio.imread(f, mode='RGBA')

        # validate spatial shape
        if img.ndim != 3 or img.shape[:2] != (tile_h, tile_w):
            raise ValueError(f'{f!s}: unexpected tile shape {img.shape}')

        # paste tile at its grid position
        y0 = int(r) * tile_h
        x0 = int(c) * tile_w
        mosaic[y0:y0 + tile_h, x0:x0 + tile_w, :] = img

    # force transparent pixels to black in RGB
    alpha0 = mosaic[..., 3] == 0
    mosaic[alpha0, :3] = background

    # drop alpha, write RGB
    rgb = mosaic[..., :3]
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(outfile, rgb, plugin='pillow')
    return outfile


class Stitcher:
    # Process-based mosaic assembler; one process per outfile

    def __init__(
            self,
            *,
            infiles: pd.Series,
            outfiles: pd.Series,
            row: pd.Series,
            col: pd.Series,
            background: int = 0,
    ):
        self.background = background

        # ensure aligned lengths
        if not (len(infiles) == len(outfiles) == len(row) == len(col)):
            raise ValueError(
                'infiles, outfiles, row and col must have the same length',
            )

        # normalize indices
        (
            self.files,
            self.group,
            self.row,
            self.col,
        ) = (
            s.reset_index(drop=True)
            for s in (infiles, outfiles, row, col)
        )

        # discover dtype from the first existing file
        self.dtype = next(
            (
                iio.imread(p).dtype
                for p in self.files
                if Path(p).is_file()
            ),
            np.uint8,
        )

        # infer tile (h, w) from the first existing file, decoding as RGBA
        sample_path = next((p for p in self.files if Path(p).is_file()), None)
        if sample_path is None:
            raise FileNotFoundError('No existing input tiles to infer shape from.')

        sample = iio.imread(sample_path, mode='RGBA')
        if sample.ndim != 3:
            raise ValueError(f'{sample_path!s}: unexpected ndim {sample.ndim}')
        self.tile_h, self.tile_w = map(int, sample.shape[:2])

        # infer mosaic (H, W) from max row/col and tile size
        max_r = int(self.row.max())
        max_c = int(self.col.max())
        self.mos_h = (max_r + 1) * self.tile_h
        self.mos_w = (max_c + 1) * self.tile_w

        # order deterministically by (group, row, col)
        order = np.lexsort((self.col.values, self.row.values, self.group.values))
        for attr in ('files', 'group', 'row', 'col'):
            setattr(self, attr, getattr(self, attr).iloc[order])

        self.unique_groups = self.group.unique()

    def run(
            self,
            *,
            max_workers: int | None = None,
    ) -> None:
        # build grouped work-list: one mosaic per unique outfile
        df = pd.DataFrame(dict(
            file=self.files.values,
            outfile=self.group.values,
            row=self.row.values.astype(int, copy=False),
            col=self.col.values.astype(int, copy=False),
        ))
        groups = df.groupby('outfile', sort=False)

        # silence multiprocessing connection/info logs that spam "Connected to: ..."
        logger = mp.util.get_logger()
        if logger.handlers:
            logger.setLevel(logging.WARNING)
        else:
            mp.util.log_to_stderr(logging.WARNING)

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            tasks: list[Future[str]] = []

            # enqueue all mosaics
            for outfile, g in tqdm(
                    groups,
                    total=groups.ngroups,
                    desc='queueing',
                    unit=f' {self.group.name}',
                    leave=False,
            ):
                fut = ex.submit(
                    _assemble_and_save,
                    g.file.tolist(),
                    g.row.to_numpy(),
                    g.col.to_numpy(),
                    outfile,
                    self.tile_h,
                    self.tile_w,
                    self.mos_h,
                    self.mos_w,
                    self.dtype,
                    background=self.background,
                )
                tasks.append(fut)

            # drain futures; propagate failures
            for fut in tqdm(
                    as_completed(tasks),
                    total=len(tasks),
                    desc='stitching',
                    unit=f' {self.group.name}',
            ):
                fut.result()
