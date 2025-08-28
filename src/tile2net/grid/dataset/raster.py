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
from .datawrapper import DataWrapper

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class RasterDataSet(
    DataSet
):

    @classmethod
    def from_tiles(
            cls,
            *,
            raster: ArrayLike,
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
        result = cls(wrapper)
        return result


class RasterDataLoader(
    DataLoader
):
    ...
