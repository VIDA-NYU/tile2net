from __future__ import annotations
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
import magicpandas as magic
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from itertools import *
from collections import UserList

class TBounds(
    UserList
):
    scale: int

    def __get__(
            self,
            instance,
            owner
    ):
        ...


    def to_scale(
            self,
            scale: int
    ) -> Self:
        ...
