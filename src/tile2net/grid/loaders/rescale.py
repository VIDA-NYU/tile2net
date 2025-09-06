from __future__ import annotations
import multiprocessing
from functools import cached_property
import cv2

from typing import *

import numpy as np
from numpy import ndarray

from .dataloader import DataLoader
# from .stitch import StitchDataSet
from .dataset import DataSet

class RescaledTile(NamedTuple):
    x0: int
    y0: int
    arr: ndarray


class RescaleDataSet(
    DataSet,
):

    def __init__(
            self,
            wrapper,
            threads: int = 1,
            scale: float = 1.,
    ):
        super().__init__(
            wrapper=wrapper,
            threads=threads,
        )
        self.scale = scale

    @cached_property
    def infile(self) -> list[str]:
        result = self.wrapper.infile.tolist()
        return result

    @cached_property
    def x0(self) -> list[int]:
        result = (
            self.wrapper.col
            .mul(self.scale)
            .astype('uint32')
            .values
            .__add__(self.wrapper.col.values)
            .tolist()
        )
        return result

    @cached_property
    def y0(self) -> list[int]:
        result = (
            self.wrapper.row
            .mul(self.scale)
            .astype('uint32')
            .values
            .__add__(self.wrapper.row.values)
            .tolist()
        )
        return result

    def __getitem__(self, item: int) -> RescaledTile:
        unscaled: ndarray = super().__getitem__(item)

        h, w, c = unscaled.shape
        h = int(h * self.scale)
        w = int(w * self.scale)
        dsize = (w, h)
        x0 = self.x0[item]
        y0 = self.y0[item]
        arr = cv2.resize(
            unscaled,
            dsize,
            interpolation=cv2.INTER_LINEAR,
        )
        result = RescaledTile(x0=x0, y0=y0, arr=arr)
        return result

    @cached_property
    def loader(self) -> RescaleDataLoader:
        num_workers = multiprocessing.cpu_count()
        result = RescaleDataLoader(
            dataset=self,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=lambda batch: batch,
            prefetch_factor=2
        )
        return result


class RescaleDataLoader(
    DataLoader
):
    if False:
        def __iter__(self) -> Iterator[RescaledTile]:
            ...

