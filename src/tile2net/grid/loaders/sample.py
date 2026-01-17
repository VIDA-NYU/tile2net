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
from .stitch import TensorDataSet, StitchDataSet

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
            static: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
            mask: Optional[ArrayLike] = None,
            force: bool = False,
            **kwargs,
    ) -> Self:
        return super().from_tiles(
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
    @cached_property
    def mask(self) -> MaskDataSet:
        """Yields the class mask for each sample"""
        raise ValueError

    @cached_property
    def raster(self) -> RasterDataSet:
        """Yields the input raster for each sample"""
        raise ValueError

    @cached_property
    def joint_transform_list(self) -> Optional[list]:
        """List of joint transforms applied to both image and label simultaneously.

        May include transforms like RandomSizeAndCrop, RandomHorizontallyFlip, 
        and RandAugment. Can be None when no joint transforms are applied.
        """
        raise ValueError

    @cached_property
    def img_transform(self) -> standard_transforms.Compose:
        """Composed transform applied to input images.

        Typically includes ToTensor() and Normalize() transforms to convert
        images to normalized tensors suitable for model input.
        """
        raise ValueError

    @cached_property
    def label_transform(self) -> extended_transforms.MaskToTensor:
        """Transform applied to label masks.

        Converts label masks to tensor format, typically using MaskToTensor
        to handle proper tensor conversion and data type casting.
        """
        raise ValueError

    @cached_property
    def skip_mask(self) -> bool:
        if (
                self.wrapper.mask
                        .isna()
                        .all()
        ):
            return True
        else:
            raise NotImplementedError
            return False

    def __getitem__(self, item) -> MiniBatch:
        img = self.raster[item]
        mask = self.mask[item]

        i = item
        scale = 1.

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                # if idx == 0 and centroid is not None:
                outputs = xform(img, mask)

                if len(outputs) == 3:
                    # img, mask, scale_float = outputs
                    img, mask, scale = outputs
                else:
                    img, mask = outputs

        if self.img_transform:
            img = self.img_transform(img)

        # todo: in the original BaseLoder.do_transforms, dump_images is between img and label transforms,
        #   but shouldn't it come after the label transform?

        if (
                mask != -1
                and self.label_transform
        ):
            mask = self.label_transform(mask)

        out = MiniBatch(
            input=img,
            label=mask,
            scale=scale,
            i=i,
        )
        return out

    @cached_property
    def target_transform(self) -> extended_transforms.MaskToTensor:
        result = extended_transforms.MaskToTensor()
        return result

    @classmethod
    def from_wrapper(
            cls,
            wrapper: SampleDataSet.__bases__[0],
            mode: str | None = None,
    ) -> Self:
        static = wrapper.static
        mask = wrapper.mask
        index = wrapper.index
        row = wrapper.row
        col = wrapper.col
        background = wrapper.background
        raster = RasterDataSet.from_tiles(
            static=static,
            index=index,
            row=row,
            col=col,
            background=background,
        )
        mask = MaskDataSet.from_tiles(
            static=mask,
            index=index,
            row=row,
            col=col,
            background=background,
        )
        out = cls(wrapper)
        out.raster = raster
        out.mask = mask
        out.mode = mode
        return out

    # @classmethod
    # def from_tiles(
    #         cls,
    #         *,
    #         raster: ArrayLike,
    #         mask: ArrayLike,
    #         i: ArrayLike,
    #         row: ArrayLike,
    #         col: ArrayLike,
    #         background: int = 0,
    # ) -> Self:
    #     wrapper = DataWrapper.from_tiles(
    #         static=raster,
    #         mask=mask,
    #         index=i,
    #         row=row,
    #         col=col,
    #         background=background,
    #     )
    #     return cls.from_wrapper(wrapper)
    #


class SampleDataLoader(
    TensorDataLoader
):
    raster: RasterDataSet


class MiniBatch(TypedDict, total=False):
    input: torch.Tensor
    label: torch.Tensor
    scale: torch.Tensor
    i: torch.Tensor
