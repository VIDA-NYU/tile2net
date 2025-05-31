from __future__ import annotations
import gdown
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

if False:
    from .tiles import Tiles


class Static:
    instance: Tiles
    owner: type[Tiles]

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    def __get__(
            self,
            instance: Tiles,
            owner: type[Tiles]
    ):
        self.instance = instance
        self.owner = owner
        return self

    @cached_property
    def path(self) -> Path:
        """Static directory into which weights are saved."""
        path = Path(__file__).parent
        while path.name != 'tile2net':
            path = path.parent
            if not path.name:
                raise FileNotFoundError('Could not find tile2net directory')
        path = path / 'static'
        return path

    def download(self):
        url = 'https://drive.google.com/drive/folders/1cu-MATHgekWUYqj9TFr12utl6VB-XKSu'
        gdown.download_folder(
            url=url,
            quiet=True,
            output=self.path.__str__(),
        )

    @property
    def hrnet_checkpoint(self) -> str:
        result = (
            self.path
            .joinpath('hrnetv2_w48_imagenet_pretrained.pth')
            .absolute().__fspath__()
        )
        return result

    @property
    def snapshot(self) -> str:
        result = (
            self.path
            .joinpath('satellite_2021.pth')
            .absolute().__fspath__()
        )
        return result
