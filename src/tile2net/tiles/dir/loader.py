
from __future__ import annotations

import dataclasses
import functools
import os
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from os import PathLike
from os import fspath
from pathlib import Path
from typing import Iterator
from typing import Self

import imageio.v3 as iio
import numpy as np
import pandas as pd
from numpy import ndarray
from toolz import curried, curry as cur, pipe

from tile2net.tiles.tiles import Tiles
from tile2net.tiles.util import returns_or_assigns
from .batchiterator import BatchIterator
class Loader:
    def __init__(
            self,
            files: pd.Series,
            group: pd.Series,
            row: pd.Series,
            col: pd.Series,
            tile_shape: tuple[int, int, int],
            mosaic_shape: tuple[int, int, int],
    ):
        if not (len(files) == len(group) == len(row) == len(col)):
            raise ValueError('files, group, row, and col must be the same length')

        self.files = files.reset_index(drop=True)
        self.group = group.reset_index(drop=True)
        self.row = row.reset_index(drop=True)
        self.col = col.reset_index(drop=True)

        self.tile_h, self.tile_w, self.tile_c = tile_shape
        self.mos_h, self.mos_w, self.mos_c = mosaic_shape

        if self.mos_h % self.tile_h or self.mos_w % self.tile_w:
            raise ValueError('mosaic dimensions are not multiples of tile size')

        # dtype discovery from first existing file
        try:
            sample_path = next(p for p in self.files if Path(p).is_file())
            self.dtype = iio.imread(sample_path).dtype
        except StopIteration:
            self.dtype = np.uint8  # default

        # sort by group, then row, then col for deterministic iteration
        order = np.lexsort((self.col, self.row, self.group))
        self.files = self.files.iloc[order]
        self.group = self.group.iloc[order]
        self.row = self.row.iloc[order]
        self.col = self.col.iloc[order]

        self.unique_groups = self.group.unique()
        self.ncols = self.mos_w // self.tile_w
        self.nrows = self.mos_h // self.tile_h

    @staticmethod
    def _read(path: str | os.PathLike) -> ndarray | None:
        try:
            if Path(path).is_file():
                return iio.imread(path)
        except Exception:
            pass
        return None

    def __iter__(self) -> Iterator[np.ndarray]:
        with ThreadPoolExecutor() as pool:
            for g in self.unique_groups:
                mask = self.group == g
                f_group = self.files[mask]
                r_group = self.row[mask].to_numpy()
                c_group = self.col[mask].to_numpy()

                out = np.zeros(
                    (self.mos_h, self.mos_w, self.mos_c),
                    dtype=self.dtype,
                )

                fut2idx = {
                    pool.submit(self._read, p): idx
                    for idx, p in enumerate(f_group)
                }

                for fut in as_completed(fut2idx):
                    img = fut.result()
                    if img is None:
                        continue
                    idx = fut2idx[fut]
                    r = r_group[idx]
                    c = c_group[idx]

                    # harmonise channel count
                    if img.ndim == 2:  # grayscale → RGB
                        img = np.repeat(img[:, :, None], self.mos_c, 2)
                    elif img.shape[2] > self.mos_c:  # drop extras (e.g. alpha)
                        img = img[:, :, : self.mos_c]
                    elif img.shape[2] < self.mos_c:  # pad missing channels
                        pad = np.zeros(
                            (*img.shape[:2], self.mos_c - img.shape[2]),
                            dtype=img.dtype,
                        )
                        img = np.concatenate((img, pad), 2)

                    y0 = r * self.tile_h
                    x0 = c * self.tile_w
                    out[
                    y0: y0 + self.tile_h,
                    x0: x0 + self.tile_w,
                    :
                    ] = img

                yield out

