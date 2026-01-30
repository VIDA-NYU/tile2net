from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import *
from typing import Union
import concurrent.futures as cf

import imageio.v3 as iio
import numpy as np
import torch.utils.data
from toolz import pipe, curried

from tile2net.grid.cfg import cfg

T = TypeVar("T", bound="StitchDataSet")
if TYPE_CHECKING:
    from .datawrapper import DataWrapper


class DataSet(
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
    ):
        self.wrapper = wrapper
        self.threads = threads

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
            .tolist()
        )
        return result

    @cached_property
    def static(self) -> list[str]:
        raise NotImplementedError

    def __getitem__(self, item):
        static = self.static[item]
        # read the sample and coerce to RGBA
        result = iio.imread(static)
        result = self._coerce_to_rgba(result)
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
