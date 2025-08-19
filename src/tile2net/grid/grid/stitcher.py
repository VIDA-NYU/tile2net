from __future__ import annotations

from typing import *

import numpy as np
from tqdm import tqdm

from concurrent.futures import Future
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import *

import imageio.v3 as iio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



# def _read_one(path: str) -> np.ndarray | None:
#     if Path(path).is_file():
#         return iio.imread(path)
#     return None
#
#
# def _assemble_and_save(  # top-level → pickle-friendly
#         files: list[str],
#         rows: np.ndarray,
#         cols: np.ndarray,
#         outfile: str,
#         tile_shape: Tuple[int, int, int],
#         mosaic_shape: Tuple[int, int, int],
#         dtype: np.dtype,
#         background: int = 0
# ) -> str:
#     """
#     Read *files*, place them into their (row, col) slots, force transparent
#     pixels to pure black, drop α, and write the finished mosaic to *outfile*.
#     """
#
#     tile_h, tile_w, _ = tile_shape
#     mos_h, mos_w, mos_c = mosaic_shape  # mos_c is **3** (RGB)
#
#     shape = (mos_h, mos_w, mos_c + 1)
#     mosaic = np.full(shape, background, dtype=dtype)
#     mosaic[..., 3] = 255  # α ← fully opaque
#
#     for f, r, c in zip(files, rows, cols, strict=True):
#         if not Path(f).is_file():  # silently skip absent grid
#             continue
#
#         # always 4 channels
#         img = iio.imread(f, mode='RGBA')
#
#         # harmonise spatial size (channel count already 4)
#         if img.ndim != 3 or img.shape[:2] != (tile_h, tile_w):
#             raise ValueError(f'{f!s}: unexpected tile shape {img.shape}')
#
#         y0, x0 = r * tile_h, c * tile_w
#         mosaic[y0:y0 + tile_h, x0:x0 + tile_w, :] = img
#
#     # ── force α-transparent pixels → black in RGB ────────────────────────────
#     alpha = mosaic[..., 3] == 0  # bool mask
#     mosaic[alpha, :3] = background  # RGB ← 0 where α == 0
#
#     # ── drop α, write RGB only ───────────────────────────────────────────────
#     rgb = mosaic[..., :3]
#     Path(outfile).parent.mkdir(parents=True, exist_ok=True)
#     iio.imwrite(outfile, rgb, plugin='pillow')  # Pillow picks format from suffix
#
#     return outfile
#
#
# class Stitcher:
#     """
#     Process-based mosaic assembler.  One **process** per outfile.
#     """
#
#     def __init__(
#             self,
#             *,
#             infiles: pd.Series,
#             outfiles: pd.Series,
#             row: pd.Series,
#             col: pd.Series,
#             tile_shape: Tuple[int, int, int],
#             mosaic_shape: Tuple[int, int, int],
#             background: int = 0,
#     ):
#         self.background = background
#
#         if not (len(infiles) == len(outfiles) == len(row) == len(col)):
#             raise ValueError(
#                 'infiles, outfiles, row and col must have the same length',
#             )
#
#         (
#             self.files,
#             self.group,
#             self.row,
#             self.col,
#         ) = (
#             s.reset_index(drop=True)
#             for s in (infiles, outfiles, row, col)
#         )
#
#         self.tile_h, self.tile_w, self.tile_c = tile_shape
#         self.mos_h, self.mos_w, self.mos_c = mosaic_shape
#
#         if (self.mos_h % self.tile_h) or (self.mos_w % self.tile_w):
#             raise ValueError('Mosaic dimensions must be multiples of tile size')
#
#         # dtype discovery (first existing file – cheap)
#         self.dtype = next(
#             (
#                 iio.imread(p).dtype
#                 for p in self.files
#                 if Path(p).is_file()
#             ),
#             np.uint8,
#         )
#
#         # sort deterministically (group, row, col)
#         order = np.lexsort((self.col, self.row, self.group))
#         for attr in ('files', 'group', 'row', 'col'):
#             setattr(self, attr, getattr(self, attr).iloc[order])
#
#         self.unique_groups = self.group.unique()
#
#     def run(
#             self,
#             *,
#             max_workers: int | None = None,
#     ) -> None:
#         """
#         Assemble and write every missing mosaic using
#         :class:`concurrent.futures.ProcessPoolExecutor`, but build the work
#         list via a single `DataFrame.groupby('outfile')` to avoid the
#         per-iteration boolean mask.
#         """
#
#         df = pd.DataFrame(dict(
#             file=self.files.values,
#             outfile=self.group.values,  # one mosaic per unique *outfile*
#             row=self.row.values,
#             col=self.col.values,
#         ))
#         groups = df.groupby('outfile', sort=False)
#
#         with ProcessPoolExecutor(max_workers=max_workers) as ex:
#             tasks: list[Future[str]] = []
#
#             # ── enqueue all mosaics --------------------------------------------
#             for outfile, g in tqdm(
#                     groups,
#                     total=groups.ngroups,
#                     desc='queueing',
#                     unit=f' {self.group.name}',
#                     leave=False,
#             ):
#                 fut = ex.submit(
#                     _assemble_and_save,
#                     g.file.tolist(),
#                     g.row.to_numpy(),
#                     g.col.to_numpy(),
#                     outfile,
#                     (self.tile_h, self.tile_w, self.tile_c),
#                     (self.mos_h, self.mos_w, self.mos_c),
#                     self.dtype,
#                     background=self.background,
#                 )
#                 # _assemble_and_save(
#                 #     files=g.file.tolist(),
#                 #     rows=g.row.to_numpy(),
#                 #     cols=g.col.to_numpy(),
#                 #     outfile=outfile,
#                 #     tile_shape=(self.tile_h, self.tile_w, self.tile_c),
#                 #     mosaic_shape=(self.mos_h, self.mos_w, self.mos_c),
#                 #     dtype=self.dtype,
#                 #     background=self.background,
#                 # )
#                 tasks.append(fut)
#
#             for fut in tqdm(
#                     as_completed(tasks),
#                     total=len(tasks),
#                     desc='stitching',
#                     unit=f' {self.group.name}',
#             ):
#                 fut.result()  # propagate failures

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
        print('⚠️AI GENERATED🤖')
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
        print('⚠️AI GENERATED🤖')
        # build grouped work-list: one mosaic per unique outfile
        df = pd.DataFrame(dict(
            file=self.files.values,
            outfile=self.group.values,
            row=self.row.values.astype(int, copy=False),
            col=self.col.values.astype(int, copy=False),
        ))
        groups = df.groupby('outfile', sort=False)

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
