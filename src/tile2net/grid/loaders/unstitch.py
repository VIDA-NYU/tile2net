from __future__ import annotations

import concurrent.futures as cf
import os
from functools import cached_property
from pathlib import Path
from typing import *

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch.utils.data

from .dataloader import BaseDataLoader
from .datawrapper import DataWrapper
from .. import frame

# todo: needs more documentation

class UnstitchDataWrapper(DataWrapper):
    """
    Wrapper for a DataFrame specifying metadata for splitting mosaics into tiles.
    """

    @frame.column
    def outfile(self) -> pd.Series:
        """output file paths for each tile"""

    @classmethod
    def from_tiles(
            cls,
            *,
            static,
            outfile,
            index,
            row,
            col,
            force: bool | Any = True,
            **kwargs,
    ) -> Self:
        data = dict(
            static=static,
            outfile=outfile,
            row=row,
            col=col,
            force=force,
            **kwargs,
        )
        df = (
            pd.DataFrame(data)
            .set_axis(index)
        )
        names = df.index.names
        by = [*names, 'row', 'col']
        df = (
            df
            .loc[lambda x: x.force]
            .reset_index()
            .sort_values(by=by)
            .set_index(names)
        )
        result: Self = cls.from_frame(df)
        return result


class UnstitchDataSet(torch.utils.data.Dataset):
    """
    Dataset for splitting mosaic images into tiles.

    Each __getitem__ call reads one mosaic, extracts all tiles belonging to
    that mosaic, and writes them to their respective output paths using
    multithreaded I/O.
    """

    def __init__(
            self,
            wrapper: UnstitchDataWrapper,
            threads: int = None,
            read: Callable[[str], np.ndarray] = None,
            write: Callable[[np.ndarray, str], None] = None,
    ):
        if threads is None:
            threads = os.cpu_count() or 4

        if read is None:
            ext = next(
                (
                    Path(f).suffix.lower()
                    for f in wrapper.static
                    if f is not None
                ),
                None,
            )
            if ext == '.npy':
                read = self.read_npy
            else:
                read = self.read_image

        self.wrapper = wrapper
        self.threads = threads
        self.read = read
        self.write = write

    def __len__(self) -> int:
        return len(self.index)

    @cached_property
    def index(self) -> list[Any]:
        """unique mosaic identifiers"""
        result = (
            self.wrapper
            .index
            .unique()
            .tolist()
        )
        return result

    @cached_property
    def grouped(self) -> dict[Any, pd.DataFrame]:
        """pre-group frame by mosaic index for fast lookup"""
        result = {
            idx: group
            for idx, group in self.wrapper.frame.groupby(
                level=self.wrapper.frame.index.names
            )
        }
        return result

    @cached_property
    def nrow(self) -> int:
        nrow = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .row
            .max()
        )
        if nrow.nunique() > 1:
            raise ValueError('mosaic rows must be identical')
        return nrow.iloc[0]

    @cached_property
    def ncol(self) -> int:
        ncol = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .col
            .max()
        )
        if ncol.nunique() > 1:
            raise ValueError('mosaic columns must be identical')
        return ncol.iloc[0]

    @cached_property
    def tile_shape(self) -> tuple[int, int]:
        """infer tile dimensions from first existing mosaic"""
        for idx in self.index:
            static = self.grouped[idx].static.iloc[0]
            if static is not None and Path(static).is_file():
                mosaic = self.read(static)
                h = mosaic.shape[0] // (self.nrow + 1)
                w = mosaic.shape[1] // (self.ncol + 1)
                return h, w
        raise FileNotFoundError('No existing mosaic to infer tile shape from.')

    @staticmethod
    def read_npy(path: str) -> np.ndarray:
        arr = np.load(path, mmap_mode=None)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return arr

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        arr = iio.imread(path)

        match arr.ndim:
            case 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            case 3:
                pass
            case _:
                raise ValueError(f'{path!s}: unsupported image ndim {arr.ndim}')

        match arr.dtype:
            case np.float32 | np.float64:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            case np.uint8:
                pass
            case _:
                arr = arr.astype(np.uint8, copy=False)

        return arr

    @staticmethod
    def write_npy(
            arr: np.ndarray,
            path: str,
    ) -> str:
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.save(outp, arr, allow_pickle=False)
        return str(outp)

    @staticmethod
    def write_image(
            arr: np.ndarray,
            path: str,
    ) -> str:
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(outp, arr)
        return str(outp)

    def extract_tile(
            self,
            mosaic: np.ndarray,
            row: int,
            col: int,
    ) -> np.ndarray:
        tile_h, tile_w = self.tile_shape
        y0 = row * tile_h
        x0 = col * tile_w
        tile = mosaic[y0:y0 + tile_h, x0:x0 + tile_w].copy()
        return tile

    def __getitem__(self, item: int) -> list[str]:
        idx = self.index[item]
        group = self.grouped[idx]

        static = group.static.iloc[0]
        mosaic = self.read(static)

        rows = group.row.tolist()
        cols = group.col.tolist()
        outfiles = group.outfile.tolist()

        def _write_tile(args: tuple[int, int, str]) -> str:
            r, c, outpath = args
            tile = self.extract_tile(mosaic, r, c)
            return self.write(tile, outpath)

        tasks = list(zip(rows, cols, outfiles, strict=True))

        if self.threads == 1:
            results = [_write_tile(t) for t in tasks]
        else:
            with cf.ThreadPoolExecutor(max_workers=self.threads) as ex:
                results = list(ex.map(_write_tile, tasks))

        return results

    def loader(
            self,
            batch_size: int = None,
            shuffle: bool = None,
            num_workers: int = None,
            cls: type = BaseDataLoader,
    ) -> BaseDataLoader:
        if batch_size is None:
            batch_size = os.cpu_count() or 4
        if shuffle is None:
            shuffle = False
        if num_workers is None:
            num_workers = 0

        out = cls(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            sampler=None,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        return out