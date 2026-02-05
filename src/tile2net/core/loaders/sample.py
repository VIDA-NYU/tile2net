from __future__ import annotations

from functools import *
from typing import *

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as standard_transforms

import tile2net.tileseg.transforms.transforms as extended_transforms
from tile2net.core import frame
from tile2net.core.loaders.image import StreamImageDataSet
from tile2net.core.loaders.mask import StreamMaskDataSet
from tile2net.core.loaders.stitch import DataWrapper
from .dataloader import TensorDataLoader
from .image import ImageDataSet
from .mask import MaskDataSet
from .stitch import StitchDataSet

if TYPE_CHECKING:
    from tile2net.core.seggrid.predict import Predict

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class SampleDataWrapper(DataWrapper):
    @classmethod
    def from_columns(
            cls,
            *,
            image_path: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
            mask: Optional[ArrayLike] = None,
            force: bool = False,
            **kwargs,
    ) -> Self:
        return super().from_columns(
            image_path=image_path,
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


class SampleDataSet(StitchDataSet):
    image: ImageDataSet
    """Yields the input image for each sample"""

    mask: MaskDataSet
    """Yields the class mask for each sample"""

    joint_transform_list: list = []
    """
    List of joint transforms applied to both image and label simultaneously.

    May include transforms like RandomSizeAndCrop, RandomHorizontallyFlip,
    and RandAugment. Can be empty when no joint transforms are applied.
    """

    img_transform: Optional[standard_transforms.Compose] = None
    """
    Composed transform applied to input images.
    
    Typically includes ToTensor() and Normalize() transforms to convert
    images to normalized tensors suitable for model input.
    """

    label_transform: Optional[extended_transforms.MaskToTensor] = None
    """
    Transform applied to label masks.
    
    Converts label masks to tensor format, typically using MaskToTensor
    to handle proper tensor conversion and data type casting.
    """

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

    def __getitem__(self, item: int) -> Sample:
        """
        Returns a Sample containing the necessary data for training or inference.

        Delegates loading static imagery to the `self.static` StaticDataSet:
            >>> self.image.__getitem__
        Delegates loading ground truth masks to the `self.mask` MaskDataSet:
            >>> self.mask.__getitem__
        See also:
            >>> Predict.__iter__
        """

        img = self.image[item]
        mask = self.mask[item]
        pred_paths = self.image.pred_path[item]
        prob_paths = self.image.prob_path[item]

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
                isinstance(mask, torch.Tensor)
                and self.label_transform
        ):
            mask = self.label_transform(mask)

        out = Sample(
            image=img,
            mask=mask,
            scale=scale,
            i=i,
            pred_paths=pred_paths,
            prob_paths=prob_paths,
        )

        unclipped_prob_path = self.image.unclipped_prob_path
        if unclipped_prob_path is not None:
            out['unclipped_prob_paths'] = unclipped_prob_path[item]

        return out

    @cached_property
    def target_transform(self) -> extended_transforms.MaskToTensor:
        result = extended_transforms.MaskToTensor()
        return result

    def __init__(
            self,
            wrapper: SampleDataWrapper,
            mode: str | None = None,
            padded_dimension: int | None = None,
            tile_dimension: int | None = None,
            *args,
            **kwargs,
    ):
        """Combines an ImageDataSet and MaskDataSet for sample loading. """
        super().__init__(
            wrapper,
            mode=mode,
            padded_dimension=padded_dimension,
            tile_dimension=tile_dimension,
            *args,
            **kwargs
        )
        self.image = ImageDataSet(
            wrapper,
            padded_dimension=padded_dimension,
            tile_dimension=tile_dimension,
            mode=mode,
            *args,
            **kwargs
        )
        # mask wrapper just uses `mask` instead of `static` as input
        wrapper = (
            wrapper.frame
            .assign(image_path=wrapper.mask)
            .pipe(wrapper.from_frame)
        )
        self.mask = MaskDataSet(
            wrapper,
            padded_dimension=padded_dimension,
            tile_dimension=tile_dimension,
            mode=mode,
            *args,
            **kwargs,
            channels=1
        )
        self.mode = mode


class StreamSampleDataSet(SampleDataSet):
    """SampleDataSet variant that handles streaming downloads with statistics tracking."""

    def __getitem__(self, item: int) -> StreamSample:
        """
        Override to handle dict returns from StreamStitchDataSet instances.
        Extracts mosaics and aggregates download statistics.
        """
        img_result = self.image[item]
        mask_result = self.mask[item]

        img = img_result['mosaic']
        mask = mask_result['mosaic']

        download_stats = {
            'success': img_result['success'] + mask_result['success'],
            'empty': img_result['empty'] + mask_result['empty'],
            'not_found': img_result['not_found'] + mask_result['not_found'],
            'failed': img_result['failed'] + mask_result['failed'],
        }

        pred_paths = self.image.pred_path[item]
        prob_paths = self.image.prob_path[item]

        i = item
        scale = 1.

        for xform in self.joint_transform_list:
            img, mask, *extras = xform(img, mask)
            if extras:
                scale = extras[0]

        if self.img_transform:
            img = self.img_transform(img)

        if (
            isinstance(mask, torch.Tensor)
            and self.label_transform
        ):
            mask = self.label_transform(mask)

        out = StreamSample(
            image=img,
            mask=mask,
            scale=scale,
            i=i,
            pred_paths=pred_paths,
            prob_paths=prob_paths,
            **download_stats,
        )

        unclipped_prob_path = self.image.unclipped_prob_path
        if unclipped_prob_path is not None:
            out['unclipped_prob_paths'] = unclipped_prob_path[item]

        return out

    def __init__(
            self,
            wrapper: SampleDataWrapper,
            tile_dimension: int,
            padding: int,
            mode: str | None = None,
            *args,
            **kwargs,
    ):
        super().__init__(
            wrapper,
            mode,
            tile_dimension=tile_dimension,
            padding=padding,
            *args,
            **kwargs,
        )
        self.image = StreamImageDataSet(
            wrapper,
            padding=padding,
            mode=mode,
            tile_dimension=tile_dimension,
            *args,
            **kwargs,
        )
        self.mask = StreamMaskDataSet(
            wrapper,
            padding=padding,
            mode=mode,
            tile_dimension=tile_dimension,
            *args,
            **kwargs,
        )



class SampleDataLoader(
    TensorDataLoader
):
    image: ImageDataSet

    if False:
        def __iter__(self) -> Iterator[Sample]:
            ...


class Sample(TypedDict, total=False):
    image: torch.Tensor
    mask: torch.Tensor
    scale: torch.Tensor
    i: torch.Tensor | int
    pred_paths: str | list[str]
    prob_paths: str | list[str]
    unclipped_prob_paths: str | list[str]


class StreamSample(Sample, total=False):
    """Sample with streaming download statistics."""
    success: int
    empty: int
    not_found: int
    failed: int
