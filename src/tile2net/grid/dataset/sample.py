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
from .datawrapper import DataWrapper

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


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
            i=i,
            row=row,
            col=col,
            background=background,
        )
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


class SampleDataLoader(
    TensorDataLoader
):
    raster: RasterDataSet


class MiniBatch(TypedDict, total=False):
    input: torch.Tensor
    label: torch.Tensor
    scale: torch.Tensor
    i: torch.Tensor
