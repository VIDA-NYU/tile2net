from __future__ import annotations

from functools import *
from typing import *

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as standard_transforms

import tile2net.tileseg.transforms.transforms as extended_transforms
from .dataloader import TensorDataLoader
from .datawrapper import DataWrapper, frame
from .mask import MaskDataSet
from .raster import RasterDataSet
from .stitch import StitchDataSet

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
    def from_columns(
            cls,
            *,
            static: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
            mask: Optional[ArrayLike] = None,
            force: bool = False,
            **kwargs,
    ) -> Self:
        return super().from_columns(
            static=static,
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
    # TensorDataSet
    StitchDataSet
):
    mask: MaskDataSet
    """Yields the class mask for each sample"""

    raster: RasterDataSet
    """Yields the input raster for each sample"""

    @cached_property
    def joint_transform_list(self) -> Optional[list]:
        """List of joint transforms applied to both image and label simultaneously.

        May include transforms like RandomSizeAndCrop, RandomHorizontallyFlip, 
        and RandAugment. Can be None when no joint transforms are applied.
        """
        return []

    @cached_property
    def img_transform(self) -> Optional[standard_transforms.Compose]:
        """Composed transform applied to input images.

        Typically includes ToTensor() and Normalize() transforms to convert
        images to normalized tensors suitable for model input.
        """
        return

    @cached_property
    def label_transform(self) -> Optional[extended_transforms.MaskToTensor]:
        """Transform applied to label masks.

        Converts label masks to tensor format, typically using MaskToTensor
        to handle proper tensor conversion and data type casting.
        """
        return

    @cached_property
    def skip_mask(self) -> bool:
        is_fully_masked = (
            self.wrapper.mask
            .isna()
            .all()
        )
        if not is_fully_masked:
            raise NotImplementedError
        return is_fully_masked

    def __getitem__(self, item) -> Sample:
        img = self.raster[item]
        mask = self.mask[item]
        pred_paths = self.raster.pred_path[item]
        prob_paths = self.raster.prob_path[item]

        i = item
        scale = 1.

        for xform in self.joint_transform_list:
            img, mask, *extras = xform(img, mask)
            if extras:
                scale = extras[0]

        if self.img_transform:
            img = self.img_transform(img)

        # todo: in the original BaseLoder.do_transforms, dump_images is between img and label transforms,
        #   but shouldn't it come after the label transform?

        if (
                mask != -1
                and self.label_transform
        ):
            mask = self.label_transform(mask)

        out = Sample(
            input=img,
            label=mask,
            scale=scale,
            i=i,
            pred_paths=pred_paths,
            prob_paths=prob_paths,
        )
        return out

    @cached_property
    def target_transform(self) -> extended_transforms.MaskToTensor:
        result = extended_transforms.MaskToTensor()
        return result

    @classmethod
    def from_wrapper(
            cls,
            wrapper: SampleDataWrapper,
            mode: str | None = None,
    ) -> Self:
        raster = RasterDataSet(wrapper)
        wrapper = (
            wrapper.frame
            .assign(static=wrapper.mask)
            .pipe(wrapper.from_frame)
        )
        mask = MaskDataSet(wrapper)
        out = cls(wrapper)
        out.raster = raster
        out.mask = mask
        out.mode = mode
        return out


class SampleDataLoader(
    TensorDataLoader
):
    raster: RasterDataSet

    if False:
        def __iter__(self) -> Iterator[Sample]:
            ...


class Sample(TypedDict, total=False):
    input: torch.Tensor
    label: torch.Tensor
    scale: torch.Tensor
    i: torch.Tensor
    pred_paths: str | list[str]
    prob_paths: str | list[str]
