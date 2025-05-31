from __future__ import annotations
import os.path
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
from .dir import Dir
import pathlib


class Probability(
    Dir
):
    @property
    def files(self) -> pd.Series:
        tiles = self.tiles
        key = self._trace
        if key in tiles:
            return tiles[key]
        else:
            format = self.format
            zoom = tiles.zoom
            it = zip(tiles.ytile, tiles.xtile)
            data = [
                format.format(z=zoom, y=ytile, x=xtile)
                for ytile, xtile in it
            ]
            result = pd.Series(data, index=tiles.index)
            tiles[key] = result
            return tiles[key]

    extension = 'npy'


class Error(
    Dir
):

    @property
    def files(self) -> pd.Series:
        tiles = self.tiles
        key = self._trace
        if key in tiles:
            return tiles[key]
        else:
            format = self.format
            zoom = tiles.zoom
            it = zip(tiles.ytile, tiles.xtile)
            data = [
                format.format(z=zoom, y=ytile, x=xtile)
                for ytile, xtile in it
            ]
            result = pd.Series(data, index=tiles.index)
            tiles[key] = result
            return tiles[key]

    extension = 'npy'


class Outdir(
    Dir
):
    @property
    def files(self):
        raise AttributeError

    # @Error
    # def error(self):
    #     ...
    #
    # @Probability
    # def probability(self):
    #     ...
    #

    # @property
    # def probability(self):
    #     tiles = self.tiles
    #     format = self.format
    #     zoom = tiles.zoom
    #     it = zip(tiles.ytile, tiles.xtile)
    #     data = [
    #         format.format(z=zoom, y=ytile, x=xtile)
    #         for ytile, xtile in it
    #     ]
    #     result = pd.Series(data, index=tiles.index, name='path')
    #     return result
    #
    # @property
    # def error(self):
    #     ...

    @Probability
    def probability(self):
        ...

    @Error
    def error(self):
        ...
