from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import *
from typing import Union

import imageio.v3 as iio
import numpy as np
import torch.utils.data
from toolz import pipe, curried

from tile2net.grid.cfg import cfg

T = TypeVar("T", bound="DataLoader")
if False:
    from .datawrapper import DataWrapper


class DataSet(
    torch.utils.data.Dataset,
):
    wrapper: DataWrapper
    """
    Data structure which reshapes the DataWrapper into a format to be
    used optimally for torch's DataLoader.
    """

    def __init__(self, wrapper: DataWrapper):
        self.wrapper = wrapper

    def __len__(self):
        """number of mosaics"""
        result = len(self.i)
        return result

    @cached_property
    def i(self) -> list[Any]:
        """mosaic identifiers; could be integer or destination path"""
        result = (
            self.wrapper
            .i
            .unique()
            .tolist()
        )
        return result

    @cached_property
    def row(self) -> list[list[int]]:
        """row within the mosaic of each tile"""
        result = (
            self.wrapper.frame
            .groupby('i', sort=False)
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
            .groupby('i', sort=False)
            .col
            .apply(list)
            .tolist()
        )
        return result

    @cached_property
    def nrow(self) -> list[int]:
        """number of rows in each mosaic"""
        result = (
            self.wrapper
            .frame
            .groupby('i', sort=False)
            .row
            .max()
            .tolist()
        )
        return result

    @cached_property
    def ncol(self) -> list[int]:
        """number of columns in each mosaic"""
        result = (
            self.wrapper
            .frame
            .groupby('i', sort=False)
            .col
            .max()
            .tolist()
        )
        return result

    @cached_property
    def infile(self) -> list[list[str]]:
        result = (
            self.wrapper
            .frame
            .groupby('i', sort=False)
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

    @staticmethod
    def _coerce_to_rgba(
            arr: np.ndarray,
    ) -> np.ndarray:
        # ensure uint8 RGBA with α=255 for opaque pixels

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)

        if arr.ndim != 3:
            raise ValueError(f'expected HxWxC array, got shape {arr.shape}')

        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        c = arr.shape[2]
        if c == 1:
            arr = np.repeat(arr, 3, axis=2)
            c = 3

        if c == 3:
            alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
            arr = np.concatenate((arr, alpha), axis=2)
        elif c != 4:
            raise ValueError(f'unsupported channel count {c}; expected 1, 3, or 4')

        result = arr
        return result

    def __getitem__(
            self,
            item,
    ) -> np.ndarray:
        # composite tiles into a single RGB mosaic

        # pull grouped lists
        files = self.infile[item]
        rows = self.row[item]
        cols = self.col[item]

        # find first present tile to infer shape
        sample_path = next(
            (f for f in files if f is not None and Path(f).is_file()),
            None,
        )
        if sample_path is None:
            raise FileNotFoundError('no readable tiles to infer shape')

        # read the sample and coerce to RGBA
        if Path(sample_path).suffix.lower() == '.npy':
            sample = np.load(sample_path, mmap_mode=None)
        else:
            sample = iio.imread(sample_path)
        sample = self._coerce_to_rgba(sample)

        # infer tile and mosaic geometry
        tile_h, tile_w = map(int, sample.shape[:2])
        grid_h = int(self.nrow[item]) + 1
        grid_w = int(self.ncol[item]) + 1
        mos_h = grid_h * tile_h
        mos_w = grid_w * tile_w

        # allocate destination RGB; background defaults to zeros
        mosaic = np.empty((mos_h, mos_w, 3), dtype=np.uint8)
        mosaic.fill(0)

        # paste the sample tile first if present in the list
        # this avoids re-reading if the sample also appears in the loop
        def _paste(
                img: np.ndarray,
                r: int,
                c: int,
        ) -> None:
            # compute region of interest
            y0 = int(r) * tile_h
            x0 = int(c) * tile_w
            sl = np.s_[y0:y0 + tile_h, x0:x0 + tile_w, :]

            # split channels and composite where alpha != 0
            rgb = img[..., :3]
            a = img[..., 3:4]
            dst = mosaic[sl]
            np.copyto(dst, rgb, where=(a != 0))

        # prime a cache for the sample to avoid duplicate decode
        cache_path = sample_path
        cache_img = sample

        # main paste loop in deterministic row/col order (already sorted)
        for f, r, c in zip(files, rows, cols, strict=True):
            if f is None:
                # skip intentionally empty tiles
                continue

            if f == cache_path:
                # reuse previously decoded sample
                _paste(cache_img, r, c)
                continue

            # read array (npy fast-path; otherwise imageio sniffing)
            if Path(f).suffix.lower() == '.npy':
                arr = np.load(f, mmap_mode=None)
            else:
                arr = iio.imread(f)

            # coerce and paste
            arr = self._coerce_to_rgba(arr)
            if arr.shape[:2] != (tile_h, tile_w):
                msg = f'unexpected tile shape {arr.shape[:2]} vs {(tile_h, tile_w)} for {f}'
                raise ValueError(msg)
            _paste(arr, r, c)

        result = mosaic
        return result


class TensorDataSet(
    DataSet
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
