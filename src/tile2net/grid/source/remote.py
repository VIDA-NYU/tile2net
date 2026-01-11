
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
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from itertools import *
from pandas.api.extensions import ExtensionArray
from .source import Source

class Remote(
    Source
):
    ...

