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

from tile2net.grid.cfg import cfg
from .dataset import TensorDataSet, T
from .dataset import DataSet
from .dataloader import DataLoader
from tile2net.tileseg.datasets.randaugment import RandAugment

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]
from .mask import MaskDataSet, MaskDataLoader
from .labels import (
    label2trainid as id_to_trainid,
    trainId2name as trainid_to_name,
)


class CentroidDataSet(
    MaskDataSet
):

    def __getitem__(self, item):
        result = super().__getitem__(item)

    @cached_property
    def coarse(self):
        return False

    @cached_property
    def custom_coarse(self):
        return False


class CentroidDataLoader(
    MaskDataLoader
):
    ...
