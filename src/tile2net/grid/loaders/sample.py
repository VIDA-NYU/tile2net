from __future__ import annotations

from functools import *
from typing import *

import numpy as np
import pandas as pd
import torch

import tile2net.tileseg.transforms.transforms as extended_transforms
from .dataloader import TensorDataLoader
from .dataset import TensorDataSet
from .datawrapper import DataWrapper, frame
from .mask import MaskDataSet
from .raster import RasterDataSet

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
            infile: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
            mask: Optional[ArrayLike] = None,
            force: bool = False,
            **kwargs,
    ) -> Self:
        return super().from_tiles(
            infile=infile,
            index=index,
            row=row,
            col=col,
            background=background,
            mask=mask,
            force=force,
            **kwargs,
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
            wrapper: SampleDataWrapper
    ) -> Self:
        raster = wrapper.infile
        mask = wrapper.mask
        index = wrapper.index
        row = wrapper.row
        col = wrapper.col
        background = wrapper.background
        raster = RasterDataSet.from_tiles(
            infile=raster,
            index=index,
            row=row,
            col=col,
            background=background,
        )
        mask = MaskDataSet.from_tiles(
            infile=mask,
            index=index,
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
            infile=raster,
            mask=mask,
            index=i,
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
