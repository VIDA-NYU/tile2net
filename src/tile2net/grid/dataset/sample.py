from __future__ import annotations

from functools import *
from typing import *

import torch
import pandas as pd
import numpy as np

import tile2net.tileseg.transforms.transforms as extended_transforms
from .dataloader import TensorDataLoader
from .dataset import TensorDataSet
from .raster import RasterDataSet
from .mask import MaskDataSet
from .datawrapper import DataWrapper, frame

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class SampleDataWrapper(
    DataWrapper
):
    @classmethod
    def from_tiles(
            cls,
            *,
            infiles: ArrayLike,
            i: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
            mask: Optional[ArrayLike] = None,
    ) -> Self:
        return super().from_tiles(
            infiles=infiles,
            i=i,
            row=row,
            col=col,
            background=background,
            mask=mask,
        )

    @frame.column
    def mask(self):
        ...



class SampleDataSet(
    TensorDataSet
):
    mask: MaskDataSet
    raster: RasterDataSet

    def __getitem__(self, item) -> MiniBatch:
        raise NotImplementedError

    @cached_property
    def target_transform(self) -> extended_transforms.MaskToTensor:
        result = extended_transforms.MaskToTensor()
        return result

    @classmethod
    def from_wrapper(
            cls,
            wrapper: DataWrapper
    ) -> Self:
        raster = wrapper.infiles
        mask = wrapper.mask
        i = wrapper.i
        row = wrapper.row
        col = wrapper.col
        background = wrapper.background
        raster = RasterDataSet.from_tiles(
            raster=raster,
            i=i,
            row=row,
            col=col,
            background=background,
        )
        mask = MaskDataSet.from_tiles(
            mask=mask,
            i=i,
            row=row,
            col=col,
            background=background,
        )
        result = cls(wrapper)
        result.raster = raster
        result.mask = mask
        return result

    @classmethod
    def from_tiles(
            cls,
            *,
            raster: ArrayLike,
            mask: ArrayLike,
            i: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
    ) -> Self:
        wrapper = DataWrapper.from_tiles(
            infiles=raster,
            mask=mask,
            i=i,
            row=row,
            col=col,
            background=background,
        )
        return cls.from_wrapper(wrapper)


class SampleDataLoader(
    TensorDataLoader
):
    raster: RasterDataSet


class MiniBatch(TypedDict, total=False):
    input: torch.Tensor
    label: torch.Tensor
    scale: torch.Tensor
    i: torch.Tensor
