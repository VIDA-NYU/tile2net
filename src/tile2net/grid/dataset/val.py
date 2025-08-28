from __future__ import annotations
from typing import NamedTuple, Optional

import torchvision.transforms as standard_transforms
from toolz import pipe, curried
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

import tile2net.tileseg.transforms.joint_transforms as joint_transforms
import tile2net.tileseg.transforms.transforms as extended_transforms
import numpy as np
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd
from functools import *
from typing import *
from types import *
from shapely import *
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from itertools import *
from pandas.api.extensions import ExtensionArray
from .dataloader import TensorDataLoader
from .datawrapper import DataWrapper
from torch.utils.data.dataloader import (
    _T_co,
    _collate_fn_t,
    _worker_init_fn_t,

)
from .sampler import DistributedSampler

from tile2net.grid.cfg import cfg
from .dataset import TensorDataSet, T
from tile2net.tileseg.datasets.randaugment import RandAugment
from .sample import SampleDataSet, SampleDataLoader


class ValDataSet(
    SampleDataSet
):

    def __getitem__(self, item):
        scale_float = 1.0
        img = self.raster[item]
        img = self.img_transform(img)
        result = dict(
            input=img,
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
        result = ValDataLoader(
            self,
            batch_size=cfg.MODEL.BS_VAL,
            num_workers=cfg.NUM_WORKERS // 2,
            shuffle=False, drop_last=False,
            sampler=self.sampler,
        )
        return result


class ValDataLoader(
    SampleDataLoader,
):
    ...
