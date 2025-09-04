from __future__ import annotations

from functools import *

import numpy as np
import torchvision.transforms as standard_transforms

from tile2net.grid.cfg import cfg
from .labels import label2trainid, trainId2name
from .sample import SampleDataSet, SampleDataLoader
from .sampler import DistributedSampler

id_to_trainid = label2trainid
trainid_to_name = trainId2name


class ValDataSet(
    SampleDataSet
):

    def __getitem__(self, item):
        scale_float = 1.0
        img = self.raster[item]
        img = self.img_transform(img)

        # only one channel
        mask = np.zeros(img.shape[1::])

        result = dict(
            input=img,
            mask=mask,
            scale=scale_float,
            i=item,
        )
        return result

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
    def name(self) -> str:
        mode = cfg.model.eval
        try:
            result = {
                None: 'val',
                'val': 'val',
                'trn': 'train',
                'folder': 'folder',
                'test': 'test',
            }.__getitem__(mode)
        except KeyError:
            raise ValueError(f'unknown eval mode {mode!r}')
        return result

    @cached_property
    def sampler(self):
        result = DistributedSampler(
            self,
            pad=False,
            permutation=False,
            consecutive_sample=False,
        )
        return result

    @cached_property
    def loader(self) -> ValDataLoader:
        ValDataLoader.__init__
        result = ValDataLoader(
            self,
            batch_size=cfg.model.bs_val,
            # num_workers=cfg.num_workers // 2,
            shuffle=False,
            drop_last=False,
            # sampler=self.sampler,
            sampler=None,

            num_workers=cfg.segmentation.num_workers,
            pin_memory=True,
            prefetch_factor=cfg.segmentation.prefetch_factor,
            persistent_workers=cfg.segmentation.persistent_workers

        )
        return result


class ValDataLoader(
    SampleDataLoader,
):
    ...
