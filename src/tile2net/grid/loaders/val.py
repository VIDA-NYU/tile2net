from __future__ import annotations

from functools import *
from typing import *

import torchvision.transforms as standard_transforms

import tile2net.tileseg.transforms.transforms as extended_transforms
from tile2net.grid.cfg import cfg
from .dataloader import BaseDataLoader
from .labels import label2trainid, trainId2name
from .sample import SampleDataSet, SampleDataLoader

id_to_trainid = label2trainid
trainid_to_name = trainId2name

class ValDataSet(
    SampleDataSet
):

    # def __getitem__(self, item):
    #     scale_float = 1.0
    #     img = self.raster[item]
    #     img = self.img_transform(img)
    #
    #     # only one channel
    #     mask = np.zeros(img.shape[1::])
    #
    #     result = dict(
    #         input=img,
    #         mask=mask,
    #         scale=scale_float,
    #         i=item,
    #     )
    #     return result

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


    # @cached_property
    # def sampler(self):
    #     result = DistributedSampler(
    #         self,
    #         pad=False,
    #         permutation=False,
    #         consecutive_sample=False,
    #     )
    #     return result


    def loader[T: BaseDataLoader](
            self,
            batch_size=None,
            shuffle=None,
            drop_last=None,
            sampler=None,
            num_workers=None,
            pin_memory=None,
            persistent_workers=None,
            worker_init_fn=None,
            cls=None,
            *args, **kwargs,
    ) -> Union[T, BaseDataLoader]:
        # Mirror datasets.setup_loaders defaults for validation loader
        if batch_size is None:
            # Prefer upper-case path, then lower-case fallback
            bs_val = cfg.model.bs_val
            if bs_val is None:
                bs_val = cfg.model.bs_val
            batch_size = bs_val
        if shuffle is None:
            shuffle = False
        if drop_last is None:
            drop_last = False
        if persistent_workers is None:
            persistent_workers = cfg.segmentation.persistent_workers
        if num_workers is None:
            num_workers = cfg.load_workers
        if cls is None:
            # default to ValDataLoader to keep type-specific behavior
            cls = ValDataLoader
        out = cls(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init_fn,
            *args, **kwargs
        )
        return out



class ValDataLoader(
    SampleDataLoader,
):
    ...
