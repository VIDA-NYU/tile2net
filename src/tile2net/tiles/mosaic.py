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
from pandas.api.extensions import ExtensionArray

import pandas as pd

if False:
    from .tiles import Tiles


class Mosaic(

):
    tiles: Tiles

    def __get__(
            self,
            instance: Tiles,
            owner: type[Tiles],
    ) -> Self:
        self.tiles = instance
        self.Tiles = owner
        return self

    @property
    def xtile(self) -> pd.Series:
        """Tile integer X of this tile in the stitched mosaic"""
        tiles = self.tiles
        if 'xtile' not in tiles.columns:
            stitched = tiles.stitched
            dscale = tiles.tscale - stitched.tscale



    @property
    def ytile(self) -> pd.Series:
        """Tile integer Y of this tile in the stitched mosaic"""

    @property
    def r(self) -> pd.Series:
        """row within the mosaic of this tile"""

    @property
    def c(self):
        """column within the mosaic of this tile"""

    @property
    def px(self):
        """Starting pixel X coordinate of this tile in the stitched mosaic"""

    @property
    def py(self):
        """Starting pixel Y coordinate of this tile in the stitched mosaic"""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        ...
