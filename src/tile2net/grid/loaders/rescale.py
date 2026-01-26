from __future__ import annotations

import multiprocessing
from functools import cached_property
from typing import *

import cv2
from numpy import ndarray

from .dataloader import DataLoader
from .dataset import DataSet


class RescaledTile(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int
    arr: ndarray


class RescaleDataSet(
    DataSet,
):

    def __init__(
            self,
            wrapper,
            threads: int = 1,
            scale: float = 1.,
            dim: int = 256,
    ):
        super().__init__(
            wrapper=wrapper,
            threads=threads,
        )
        self.scale = scale
        self.dim = dim

    @cached_property
    def static(self) -> list[str]:
        result = self.wrapper.image_path.tolist()
        return result

    @cached_property
    def x0(self) -> list[int]:
        result = (
            self.wrapper.col
            .mul(self.dim + 1)
            .mul(self.scale)
            .round()
            .astype('uint32')
            .values
            .tolist()
        )
        return result

    @cached_property
    def y0(self) -> list[int]:
        result = (
            self.wrapper.row
            .mul(self.dim + 1)
            .mul(self.scale)
            .round()
            .astype('uint32')
            .values
            .tolist()
        )
        return result

    @cached_property
    def x1(self) -> list[int]:
        result = (
            self.wrapper.col
            .mul(self.dim + 1)
            .add(self.dim)
            .mul(self.scale)
            .round()
            .astype('uint32')
            .values
            .tolist()
        )
        return result

    @cached_property
    def y1(self) -> list[int]:
        result = (
            self.wrapper.row
            .mul(self.dim + 1)
            .add(self.dim)
            .mul(self.scale)
            .round()
            .astype('uint32')
            .values
            .tolist()
        )
        return result

    # @cached_property
    # def w(self):
    #     result = [
    #         x1 - x0
    #         for x0, x1 in zip(self.x0, self.x1)
    #     ]
    #     return result
    #
    # @cached_property
    # def h(self):
    #     result = [
    #         y1 - y0
    #         for y0, y1 in zip(self.y0, self.y1)
    #     ]
    #     return result

    def __getitem__(self, item: int) -> RescaledTile:
        unscaled: ndarray = super().__getitem__(item)

        x0 = self.x0[item]
        y0 = self.y0[item]
        x1 = self.x1[item]
        y1 = self.y1[item]
        dsize = (x1 - x0, y1 - y0)
        if self.scale < 1:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_CUBIC
        arr = cv2.resize(unscaled, dsize, interpolation=inter)
        result = RescaledTile(x0=x0, y0=y0, arr=arr, x1=x1, y1=y1)
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
