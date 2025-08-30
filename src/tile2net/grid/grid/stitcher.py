
from __future__ import annotations

from collections import deque
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from pathlib import Path
import os

import imageio.v3 as iio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# tile readers ---------------------------------------------------------------

def _read_image_to_rgba(path: str) -> np.ndarray:
    # read any image format, coerce to RGBA uint8
    arr = iio.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f'{path!s}: unsupported image ndim {arr.ndim}')
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 3:
        pad = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate((arr, pad), axis=2)
    elif arr.shape[2] != 4:
        raise ValueError(f'{path!s}: expected channels in (1,3,4), got {arr.shape[2]}')
    return arr


def _read_npy(path: str) -> np.ndarray:
    # expect HxWxC uint8; no decoding
    arr = np.load(path, mmap_mode=None)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f'{path!s}: expected HxWxC array, got {arr.shape}')
    # ensure contiguous uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    # synthesize RGBA if needed (α=255)
    if arr.shape[2] == 3:
        pad = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate((arr, pad), axis=2)
    return arr


def _read_tiles_threaded(
        files: list[str],
        *,
        reader_fn,
        max_workers: int,
) -> list[np.ndarray]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(reader_fn, files, chunksize=4))


# worker ---------------------------------------------------------------------

def _assemble_and_save(  # top-level → pickle-friendly
        files: list[str],
        rows: np.ndarray,
        cols: np.ndarray,
        outfile: str,
        tile_h: int,
        tile_w: int,
        mos_h: int,
        mos_w: int,
        background: int = 0,
        *,
        thread_reads: bool = True,
        max_read_threads: int = 2,    # keep small to avoid I/O oversubscription
        png_compress_level: int = 0,  # fastest
) -> str:
    # single-point existence filter (avoid re-checks in the hot loop)
    # note: we do not stat() again in the paste loop
    present = [
        (int(r) * tile_h, int(c) * tile_w, f)
        for f, r, c in zip(files, rows, cols, strict=True)
        if Path(f).is_file()
    ]

    # allocate RGB mosaic as uint8 (decode/read paths feed uint8)
    mosaic = np.empty((mos_h, mos_w, 3), dtype=np.uint8)
    if background == 0:
        mosaic.fill(0)
    else:
        mosaic[...] = background

    if present:
        f_list = [f for _, _, f in present]
        # choose reader by the first file's extension
        first_ext = Path(f_list[0]).suffix.lower().lstrip('.')
        reader_fn = _read_npy if first_ext == 'npy' else _read_image_to_rgba

        # parallelize reads inside the worker (small thread count)
        if thread_reads and len(f_list) > 3:
            cpu_half = max(1, (os.cpu_count() or 2) // 2)
            max_threads = min(max_read_threads, len(f_list), cpu_half)
            imgs = _read_tiles_threaded(
                f_list,
                reader_fn=reader_fn,
                max_workers=max_threads,
            )
        else:
            imgs = [reader_fn(f) for f in f_list]

        # single-pass paste with per-tile alpha mask
        for (y0, x0, _), img in zip(present, imgs, strict=True):
            if img.ndim != 3 or img.shape[:2] != (tile_h, tile_w):
                raise ValueError(f'unexpected tile shape {img.shape} vs {(tile_h, tile_w)}')

            rgb = img[..., :3]
            a = img[..., 3:4]  # keep dims for broadcasting
            sl = np.s_[y0:y0 + tile_h, x0:x0 + tile_w, :]
            dst = mosaic[sl]
            np.copyto(dst, rgb, where=(a != 0))

    # write result (NPY has special handling; otherwise use default image writer)
    outp = Path(outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_ext = outp.suffix.lower().lstrip('.')
    if out_ext == 'npy':
        np.save(str(outp), mosaic, allow_pickle=False)
        return str(outp)
    else:
        if out_ext == 'png':
            iio.imwrite(
                str(outp),
                mosaic,
                plugin='pillow',
                compress_level=png_compress_level,
                optimize=False,
            )
        else:
            iio.imwrite(str(outp), mosaic)
        return str(outp)


# orchestrator ---------------------------------------------------------------

class Stitcher:
    # Process-based mosaic assembler; bounded in-flight jobs; optional npy IO

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
            raise ValueError('infiles, outfiles, row and col must have the same length')

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

        # infer tile (h, w) via a sample consistent with first present input extension
        sample_path = next((p for p in self.files if Path(p).is_file()), None)
        if sample_path is None:
            raise FileNotFoundError('No existing input tiles to infer shape from.')
        sample_ext = Path(sample_path).suffix.lower().lstrip('.')
        sample_reader = _read_npy if sample_ext == 'npy' else _read_image_to_rgba
        sample = sample_reader(sample_path)
        if sample.ndim != 3:
            raise ValueError(f'{sample_path!s}: unexpected ndim {sample.ndim}')
        self.tile_h, self.tile_w = map(int, sample.shape[:2])
        self.dtype = np.uint8  # IO path is uint8; keep explicit

        # infer mosaic (H, W) from global max row/col and tile size
        max_r = int(self.row.max())
        max_c = int(self.col.max())
        self.mos_h = (max_r + 1) * self.tile_h
        self.mos_w = (max_c + 1) * self.tile_w

        # deterministically order by (group, row, col)
        order = np.lexsort((self.col.values, self.row.values, self.group.values))
        for attr in ('files', 'group', 'row', 'col'):
            setattr(self, attr, getattr(self, attr).iloc[order])

        self.unique_groups = self.group.unique()

        # performance knobs (keep conservative to avoid I/O thrash)
        self._thread_reads = True
        self._max_read_threads = 2       # 2–4 recommended; keep small when writing PNG
        self._png_compress_level = 0     # fastest
        self._inflight_factor = 2        # ~2× workers in flight

    def run(
            self,
            *,
            max_workers: int | None = None,
    ) -> None:
        # choose a conservative default if not provided (single NVMe ~4–8)
        if max_workers is None:
            max_workers = 2

        # build grouped work-list: one mosaic per unique outfile
        df = pd.DataFrame(dict(
            file=self.files.values,
            outfile=self.group.values,
            row=self.row.values.astype(int, copy=False),
            col=self.col.values.astype(int, copy=False),
        ))
        groups = df.groupby('outfile', sort=False)

        # prepare a job tuple list to feed the bounded submitter
        jobs: list[tuple] = []
        for outfile, g in groups:
            jobs.append((
                _assemble_and_save,
                g.file.tolist(),
                g.row.to_numpy(),
                g.col.to_numpy(),
                outfile,
                self.tile_h,
                self.tile_w,
                self.mos_h,
                self.mos_w,
                self.background,
            ))

        inflight_limit = max(2, self._inflight_factor * max_workers)

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            # producer/consumer with back-pressure
            pending: deque[Future] = deque()
            it = iter(jobs)

            # prime window
            for _ in range(min(inflight_limit, len(jobs))):
                args = next(it, None)
                if args is None:
                    break
                fut = ex.submit(
                    args[0],                 # _assemble_and_save
                    *args[1:],
                    thread_reads=self._thread_reads,
                    max_read_threads=self._max_read_threads,
                    png_compress_level=self._png_compress_level,
                )
                pending.append(fut)

            # progress bar over total jobs
            pbar = tqdm(total=len(jobs), desc='stitching', unit=' job')

            # drain while maintaining the window
            completed = 0
            while pending:
                fut = pending.popleft()
                fut.result()  # propagate error if any
                completed += 1
                pbar.update(1)

                args = next(it, None)
                if args is not None:
                    nxt = ex.submit(
                        args[0],
                        *args[1:],
                        thread_reads=self._thread_reads,
                        max_read_threads=self._max_read_threads,
                        png_compress_level=self._png_compress_level,
                    )
                    pending.append(nxt)

            pbar.close()
