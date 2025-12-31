from __future__ import annotations
from . import datawrapper
import os

import concurrent.futures as cf
from functools import cached_property
from pathlib import Path
from typing import *
from typing import Union
import logging
import imageio.v3 as iio
from .dataloader import BaseDataLoader
import numpy as np
import torch.utils.data
from toolz import pipe, curried

from tile2net.grid.cfg import cfg
from tile2net.grid.frame import frame



def _worker_init(_: int) -> None:
    logging.disable(logging.ERROR)
    os.environ['TQDM_DISABLE'] = '1'
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)


class DataWrapper(datawrapper.DataWrapper):
    @frame.column
    def mask(self):
        ...



class StitchDataSet(
    torch.utils.data.Dataset,
):
    wrapper: DataWrapper
    """
    Data structure which reshapes the DataWrapper into a format to be
    used optimally for torch's DataLoader.
    """

    def __init__(
            self,
            wrapper: DataWrapper,
            threads: int = 1,
            read: Callable[[str], np.ndarray] = None,
            write: Callable[[np.ndarray, str], None] = None,
            *args,
            **kwargs,
    ):
        if read is None:
            # Filter out None values when looking for file extension
            ext = next(
                (
                    Path(file)
                    .suffix
                    .lower()
                    for file in wrapper.infile
                    if file is not None
                ),
                None
            )
            if ext == '.npy':
                read = self.read_npy
            else:
                read = self.read_image

        self.wrapper = wrapper
        self.threads = threads
        self.read = read
        self.write = write

    def __len__(self):
        """number of mosaics"""
        result = len(self.index)
        return result

    @cached_property
    def index(self) -> list[Any]:
        """mosaic identifiers; could be integer or destination path"""
        result = (
            self.wrapper
            .index
            .unique()
        )
        return result

    @cached_property
    def row(self) -> list[list[int]]:
        """row within the mosaic of each tile"""
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names)
            .row
            .apply(list)
            .tolist()
        )
        return result

    @cached_property
    def col(self) -> list[list[int]]:
        """column within the mosaic of each tile"""
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names, )
            .col
            .apply(list)
            .tolist()
        )
        return result

    @cached_property
    def nrow(self) -> int:
        """number of rows in each mosaic"""
        nrow = (
            self.wrapper
            .frame
            .groupby(level=self.wrapper.frame.index.names)
            .row
            .max()
        )
        if nrow.nunique() > 1:
            raise ValueError('mosaic rows must be identical')
        return nrow.iloc[0]

    @cached_property
    def ncol(self) -> int:
        """number of columns in each mosaic"""
        ncol = (
            self.wrapper
            .frame
            .groupby(level=self.wrapper.frame.index.names)
            .col
            .max()
        )
        if ncol.nunique() > 1:
            raise ValueError('mosaic columns must be identical')
        return ncol.iloc[0]

    @cached_property
    def infile(self) -> list[list[str]]:
        result = (
            self.wrapper
            .frame
            .groupby(level=self.wrapper.frame.index.names, )
            .infile
            .apply(list)
            .tolist()
        )
        return result

    @cached_property
    def crop_size(self) -> Union[int, Tuple[int, int]]:
        result = cfg.dataset.crop_size
        if (
                isinstance(result, str)
                and ',' in result
        ):
            result = pipe(
                result.split(','),
                curried.map(int),
                list
            )
        elif isinstance(result, (list, tuple)):
            ...
        else:
            result = int(result)
        return result

    @cached_property
    def mean_std(self) -> tuple[list[float], list[float]]:
        mean = cfg.dataset.mean
        std = cfg.dataset.std
        result = (mean, std)
        return result

    @cached_property
    def sample(self) -> np.ndarray:
        try:
            path: str = next(
                path
                for paths in self.infile
                for path in paths
                if Path(path).is_file()
            )
        except StopIteration:
            raise FileNotFoundError('No existing input tiles to infer shape from.')
        except Exception:
            raise
        sample = self.read(path)
        if sample.ndim != 3:
            raise ValueError(f'{path!s}: unexpected ndim {sample.ndim}')
        return sample

    @staticmethod
    def read_npy(path: str) -> np.ndarray:
        arr = np.load(path, mmap_mode=None)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f'{path!s}: expected HxWxC array, got {arr.shape}')
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if arr.shape[2] == 3:
            pad = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
            arr = np.concatenate((arr, pad), axis=2)
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

        match arr.shape[2]:
            case 1:
                arr = np.repeat(arr, 3, axis=2)
            case 3:
                pad = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
                arr = np.concatenate((arr, pad), axis=2)
            case 4:
                pass
            case _:
                raise ValueError(f'{path!s}: expected channels in (1,3,4), got {arr.shape[2]}')

        return arr

    # @staticmethod
    # def _coerce_to_rgba(
    #         arr: np.ndarray,
    # ) -> np.ndarray:
    #     # ensure uint8 RGBA with α=255 for opaque pixels
    #
    #     if arr.ndim == 2:
    #         arr = np.repeat(arr[..., None], 3, axis=2)
    #
    #     if arr.ndim != 3:
    #         raise ValueError(f'expected HxWxC array, got shape {arr.shape}')
    #
    #     if arr.dtype in (np.float32, np.float64):
    #         arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)
    #     elif arr.dtype != np.uint8:
    #         arr = arr.astype(np.uint8, copy=False)
    #
    #     c = arr.shape[2]
    #     if c == 1:
    #         arr = np.repeat(arr, 3, axis=2)
    #         c = 3
    #
    #     if c == 3:
    #         alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
    #         arr = np.concatenate((arr, alpha), axis=2)
    #     elif c != 4:
    #         raise ValueError(f'unsupported channel count {c}; expected 1, 3, or 4')
    #
    #     result = arr
    #     return result
    #
    # def __getitem__(
    #         self,
    #         item,
    # ) -> np.ndarray:
    #     # composite tiles into a single RGB mosaic
    #
    #     # pull grouped lists
    #     files = self.infile[item]
    #     rows = self.row[item]
    #     cols = self.col[item]
    #
    #     # find first present tile to infer shape
    #     sample_path = next(
    #         (f for f in files if f is not None and Path(f).is_file()),
    #         None,
    #     )
    #     if sample_path is None:
    #         raise FileNotFoundError('no readable tiles to infer shape')
    #
    #     # read the sample and coerce to RGBA
    #     if Path(sample_path).suffix.lower() == '.npy':
    #         sample = np.load(sample_path, mmap_mode=None)
    #     else:
    #         sample = iio.imread(sample_path)
    #     sample = self._coerce_to_rgba(sample)
    #
    #     # infer tile and mosaic geometry
    #     tile_h, tile_w = map(int, sample.shape[:2])
    #     grid_h = int(self.nrow[item]) + 1
    #     grid_w = int(self.ncol[item]) + 1
    #     mos_h = grid_h * tile_h
    #     mos_w = grid_w * tile_w
    #
    #     # allocate destination RGB; background defaults to zeros
    #     mosaic = np.empty((mos_h, mos_w, 3), dtype=np.uint8)
    #     mosaic.fill(0)
    #
    #     # paste the sample tile first if present in the list
    #     # this avoids re-reading if the sample also appears in the loop
    #     def _paste(
    #             img: np.ndarray,
    #             r: int,
    #             c: int,
    #     ) -> None:
    #         # compute region of interest
    #         y0 = int(r) * tile_h
    #         x0 = int(c) * tile_w
    #         sl = np.s_[y0:y0 + tile_h, x0:x0 + tile_w, :]
    #
    #         # split channels and composite where alpha != 0
    #         rgb = img[..., :3]
    #         a = img[..., 3:4]
    #         dst = mosaic[sl]
    #         np.copyto(dst, rgb, where=(a != 0))
    #
    #     # prime a cache for the sample to avoid duplicate decode
    #     cache_path = sample_path
    #     cache_img = sample
    #
    #     # main paste loop in deterministic row/col order (already sorted)
    #     # optionally pre-read tiles in parallel to accelerate I/O
    #     if self.threads != 1:
    #         # submit read tasks for each tile except None and cached sample
    #         tasks: list[tuple[int, str]] = [
    #             (i, f)
    #             for i, f in enumerate(files)
    #             if f is not None and f != cache_path
    #         ]
    #
    #         def _read(path: str) -> np.ndarray:
    #             if Path(path).suffix.lower() == '.npy':
    #                 arr = np.load(path, mmap_mode=None)
    #             else:
    #                 arr = iio.imread(path)
    #             return self._coerce_to_rgba(arr)
    #
    #         results: dict[int, np.ndarray] = {}
    #         # cap workers at positive int; if threads is 0 or negative, fallback to default behavior
    #         max_workers = self.threads if self.threads and self.threads > 0 else None
    #         with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
    #             futs = {
    #                 ex.submit(_read, f): i
    #                 for i, f in tasks
    #             }
    #             for fut in cf.as_completed(futs):
    #                 idx = futs[fut]
    #                 results[idx] = fut.result()
    #
    #         # sequentially paste following the original order
    #         for i, (f, r, c) in enumerate(zip(files, rows, cols, strict=True)):
    #             if f is None:
    #                 continue
    #             if f == cache_path:
    #                 _paste(cache_img, r, c)
    #                 continue
    #             arr = results[i]
    #             if arr.shape[:2] != (tile_h, tile_w):
    #                 msg = f'unexpected tile shape {arr.shape[:2]} vs {(tile_h, tile_w)} for {f}'
    #                 raise ValueError(msg)
    #             _paste(arr, r, c)
    #     else:
    #         for f, r, c in zip(files, rows, cols, strict=True):
    #             if f is None:
    #                 # skip intentionally empty tiles
    #                 continue
    #
    #             if f == cache_path:
    #                 # reuse previously decoded sample
    #                 _paste(cache_img, r, c)
    #                 continue
    #
    #             # read array (npy fast-path; otherwise imageio sniffing)
    #             if Path(f).suffix.lower() == '.npy':
    #                 arr = np.load(f, mmap_mode=None)
    #             else:
    #                 arr = iio.imread(f)
    #
    #             # coerce and paste
    #             arr = self._coerce_to_rgba(arr)
    #             if arr.shape[:2] != (tile_h, tile_w):
    #                 msg = f'unexpected tile shape {arr.shape[:2]} vs {(tile_h, tile_w)} for {f}'
    #                 raise ValueError(msg)
    #             _paste(arr, r, c)
    #
    #     result = mosaic
    #     return result

    @cached_property
    def h(self):
        return (self.nrow + 1) * self.sample.shape[0]

    @cached_property
    def w(self):
        return (self.ncol + 1) * self.sample.shape[1]

    @property
    def mosaic(self):
        out = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        out.fill(0)
        return out

    def __getitem__(
            self,
            item,
    ) -> np.ndarray:
        # composite tiles into a single RGB mosaic

        # pull grouped lists
        files = self.infile[item]
        rows = self.row[item]
        cols = self.col[item]
        index = self.index[item]
        mosaic = self.mosaic

        if self.threads == 1:
            for f, r, c in zip(files, rows, cols, strict=True):
                if f is None:
                    # skip empty tiles
                    continue
                arr = self.read(f)
                self.paste(mosaic, arr, r, c)

        else:
            # submit read tasks for each tile except None
            tasks: list[tuple[int, str]] = [
                (i, f)
                for i, f in enumerate(files)
                if f is not None
            ]

            results: dict[int, np.ndarray] = {}
            max_workers = self.threads

            with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(self.read, f): i
                    for i, f in tasks
                }
                for fut in cf.as_completed(futs):
                    idx = futs[fut]
                    results[idx] = fut.result()

            # sequentially paste following the original order
            for i, (f, r, c) in enumerate(zip(files, rows, cols, strict=True)):
                if f is None:
                    continue
                arr = results[i]
                self.paste(mosaic, arr, r, c)

        if self.write:
            out = self.write(mosaic, index)
        else:
            out = mosaic
        return out

    def paste(
            self,
            mosaic: np.ndarray,
            tile: np.ndarray,
            row: int,
            col: int,
    ) -> None:
        """
        Paste a tile into the mosaic at the given row/column position.
        Uses alpha channel compositing: only pixels where alpha != 0 are copied.

        Based on the original _paste implementation from __getitem__.
        """
        tile_h = tile.shape[0]
        tile_w = tile.shape[1]

        # compute region of interest
        y0 = row * tile_h
        x0 = col * tile_w
        sl = np.s_[y0:y0 + tile_h, x0:x0 + tile_w, :]

        # split channels and composite where alpha != 0
        rgb = tile[..., :3]
        a = tile[..., 3:4]
        dst = mosaic[sl]
        np.copyto(dst, rgb, where=(a != 0))

    @staticmethod
    def write_npy(
            arr: np.ndarray,
            file: str,
    ):
        outp = Path(file)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.save(outp, arr, allow_pickle=False)
        return str(outp)

    @staticmethod
    def write_image(
            arr: np.ndarray,
            file: str,
    ):
        outp = Path(file)
        outp.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(outp, arr)
        return str(outp)

    def loader[T: BaseDataLoader](
            self,
            batch_size=None,
            shuffle=None,
            drop_last=None,
            sampler=None,
            num_workers=None,
            pin_memory=None,
            persistent_workers=None,
            worker_init_fn=None,
            cls=BaseDataLoader
    ) -> Union[T, BaseDataLoader]:
        if batch_size is None:
            batch_size = os.cpu_count()
        if shuffle is None:
            shuffle = False
        if drop_last is None:
            drop_last = False
        if sampler is None:
            sampler = None
        if num_workers is None:
            num_workers = 0
        if pin_memory is None:
            pin_memory = True
        if persistent_workers is None:
            persistent_workers = False

        out = cls(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init_fn,
        )
        return out


class TensorDataSet(
    StitchDataSet
):
    """
    DataSet extension which returns torch.Tensor from __getitem__
    """

    def __getitem__(
            self,
            item,
    ) -> torch.Tensor:
        arr = super().__getitem__(item)
        result = (
            torch
            .from_numpy(arr)
            .permute(2, 0, 1)
            .contiguous()
        )
        return result
