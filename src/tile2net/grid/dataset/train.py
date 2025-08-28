from __future__ import annotations

from functools import *
from typing import *

import torchvision.transforms as standard_transforms

import tile2net.tileseg.transforms.joint_transforms as joint_transforms
import tile2net.tileseg.transforms.transforms as extended_transforms
from tile2net.grid.cfg import cfg
from tile2net.tileseg.datasets.randaugment import RandAugment
from .mask import MaskDataSet
from .raster import RasterDataSet
from .sampler import DistributedSampler
from .sample import MiniBatch
from .sample import SampleDataSet, SampleDataLoader


class TrainDataSet(
    SampleDataSet
):
    """ SampleDataSet for training """
    mask: MaskDataSet
    # todo: centroid, epoch, sampling
    raster: RasterDataSet
    centroid: dict

    def __getitem__(self, item) -> MiniBatch:
        scale_float = 1.
        img = self.raster[item]
        mask = self.mask[item]
        centroid = self.centroid[item]
        img, mask, centroid = self.random_size_and_crop(img, mask, centroid)
        img, mask = self.random_horizontally_flip(img, mask)
        img, mask = self.rand_augment(img, mask)
        img = self.img_transform(img)
        mask = self.target_transform(mask)
        result = dict(
            input=img,
            label=mask,
            scale=scale_float,
            i=item,
        )
        return result

    @cached_property
    def img_transform(self) -> standard_transforms.Compose:
        items = []
        if cfg.model.color_aug:
            jitter = extended_transforms.ColorJitter(
                brightness=cfg.model.color_aug,
                contrast=cfg.model.color_aug,
                saturation=cfg.model.color_aug,
                hue=cfg.model.color_aug,
            )
            items.append(jitter)
        if cfg.model.bblur:
            item = extended_transforms.RandomBilateralBlur()
            items.append(item)
        elif cfg.model.gblur:
            items.append(extended_transforms.RandomGaussianBlur())
        mean, std = self.mean_std
        items.append(standard_transforms.ToTensor())
        items.append(standard_transforms.Normalize(mean, std))
        result = standard_transforms.Compose(items)
        return result

    @cached_property
    def joint_transform_list(self) -> list[
        Union[
            joint_transforms.RandomSizeAndCrop,
            joint_transforms.RandomHorizontallyFlip,
            RandAugment,
        ]
    ]:
        result = []
        item = joint_transforms.RandomSizeAndCrop(
            self.crop_size,
            False,
            scale_min=cfg.model.scale_min,
            scale_max=cfg.model.scale_max,
            full_size=cfg.model.full_crop_modeling,
            pre_size=cfg.model.pre_size
        )

        result.append(item)

        item = joint_transforms.RandomHorizontallyFlip()
        result.append(item)

        if cfg.model.rand_augment is not None:
            N, M = [
                int(i)
                for i in cfg.model.rand_augment.split(',')
            ]
            assert (
                    isinstance(N, int)
                    and isinstance(M, int)
            ), f'Either N {N} or M {M} not integer'
            item = RandAugment(N, M)
            result.append(item)

        return result

    @cached_property
    def random_size_and_crop(self):
        return joint_transforms.RandomSizeAndCrop(
            self.crop_size,
            False,
            scale_min=cfg.model.scale_min,
            scale_max=cfg.model.scale_max,
            full_size=cfg.model.full_crop_modeling,
            pre_size=cfg.model.pre_size
        )

    @cached_property
    def random_horizontally_flip(self):
        return joint_transforms.RandomHorizontallyFlip()

    @cached_property
    def rand_augment(self):
        if cfg.model.rand_augment is None:
            return None
        N, M = [
            int(i)
            for i in cfg.model.rand_augment.split(',')
        ]
        assert (
                isinstance(N, int)
                and isinstance(M, int)
        ), f'Either N {N} or M {M} not integer'
        return RandAugment(N, M)

    @cached_property
    def sampler(self):
        result = DistributedSampler(
            self,
            pad=True,
            permutation=True,
            consecutive_sample=True,
        )
        return result


class TrainDataLoader(
    SampleDataLoader
):
    # todo: centroid, epoch
    # todo: incorporate both mask and satellite loaders
    """SampleDataLoader for training"""
