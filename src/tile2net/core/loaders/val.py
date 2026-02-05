from __future__ import annotations

from functools import *

import torchvision.transforms as standard_transforms

import tile2net.tileseg.transforms.transforms as extended_transforms
from tile2net.core.cfg import cfg
from tile2net.core.loaders.sample import StreamSampleDataSet
from .sample import SampleDataSet


class ValDataSet(
    SampleDataSet
):

    @cached_property
    def img_transform(self) -> standard_transforms.Compose:
        mean = cfg.dataset.mean
        std = cfg.dataset.std
        items = []
        items.append(standard_transforms.ToTensor())
        items.append(standard_transforms.Normalize(mean, std))
        result = standard_transforms.Compose(items)
        return result

    @cached_property
    def label_transform(self):
        return extended_transforms.MaskToTensor()

    @cached_property
    def joint_transform_list(self):
        return []

    @cached_property
    def sampler(self):
        msg = f'Distributed inference is not implemented yet.'
        raise NotImplementedError(msg)
        result = DistributedSampler(
            self,
            pad=False,
            permutation=False,
            consecutive_sample=False,
        )
        return result


class StreamValDataSet(
    StreamSampleDataSet,
    ValDataSet,
):
    """ValDataSet variant for streaming with download statistics."""

