from __future__ import annotations
from concurrent.futures import *

import copy
from functools import cached_property
from types import SimpleNamespace

import matplotlib.pyplot as plt
import cv2

import os.path

import shapely

from tile2net.raster.tile import Tile
from pathlib import Path
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
from collections import *
from tile2net.raster import Raster
from types import *
from types import *
from typing import *
# from tile2net.logger import logger
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# from tile2net.raster import Raster
#
from typing import TypedDict


class Surface(UserDict, SimpleNamespace):
    path: str
    usecols: list[str]
    col: Union[int, dict[str, list[str]]]
    color: str
    order: int


class Config(dict):
    parkinglot: Surface
    road: Surface
    median: Surface
    alley: Surface

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        surfaces = {
            key: Surface(**value)
            for key, value in d.items()
        }
        return cls(**surfaces)

    def __getattr__(self, item) -> Surface:
        return self[item]


class Mask:
    def __init__(
            self,
            raster: Raster,
            geometry: GeoDataFrame,
            config,
    ):
        # todo: use property setters to define these
        self.geometry = geometry
        self.raster = raster
        self.config = config

    @property
    def geometry(self):
        ...

    @geometry.setter
    def geometry(self, value):
        # todo: gdf or file
        ...

    @property
    def config(self) -> Config:
        ...

    @config.setter
    def config(self, value):
        # todo: json or dict
        ...

    @cached_property
    def tiles(self) -> GeoDataFrame:
        """GeoSeries of tile bboxes"""
        tiles: list[Tile] = self.raster.tiles.tolist()
        count = len(tiles)

        # this is one place where tile2net will benefit from
        # a magicpandas refactor; currently we are iterating
        # and calling getattr on every tile 🐌
        def fromiter(side: str, dtype=float) -> ndarray:
            return np.fromiter(
                (getattr(tile, side) for tile in tiles),
                dtype=dtype,
                count=count,
            )

        n = fromiter('top')
        s = fromiter('bottom')
        w = fromiter('left')
        e = fromiter('right')
        iy = fromiter('y', dtype=int)
        ix = fromiter('x', dtype=int)
        itiles = fromiter('idd', dtype=int)

        geometry = shapely.box(w, s, e, n)
        index = pd.Index(itiles, name='itile')
        filename = Series((
            f'{iy_}_{ix_}_{itile}'
            for iy_, ix_, itile in zip(iy, ix, itiles)
        ), dtype='string')
        result = GeoDataFrame({
            'geometry': geometry,
            'ix': ix,
            'iy': iy,
            'filename': filename,
        }, index=index)
        return result

    @cached_property
    def clippings(self) -> GeoDataFrame:
        # clip the geometry to the tiles
        matches: GeoDataFrame = (
            self
            .geometry
            .sjoin(self.tiles, how='inner', op='intersects')
            .rename(columns={'index_right': self.tiles.index.name})
        )
        tiles = self.tiles.loc[matches.index]
        matches.geometry = matches.intersection(tiles.geometry, align=False)
        return matches

    @classmethod
    def from_file(
            cls,
            raster: Raster,
            path: str | Path,
    ) -> Self:
        path = Path(path)
        match path.suffix:
            case '.parquet':
                geometry = gpd.read_parquet(path)
            case '.feather':
                geometry = gpd.read_feather(path)
            case _:
                geometry = gpd.read_file(path)
        return cls(raster, geometry)

    @cached_property
    def figax(self) -> tuple[plt.Figure, plt.Axes]:
        img = self.raster.black.array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dpi = 1200
        fig, ax = plt.subplots(
            figsize=((img.shape[0] / float(dpi)), (img.shape[1] / float(dpi))))
        plt.box(False)
        fig.dpi = dpi
        fig.tight_layout(pad=0)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.margins(0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_facecolor('black')

        return fig, ax

    def to_directory(self, path: str | Path = None):
        outdir = os.path.join(path, 'annotations')
        os.makedirs(outdir, exist_ok=True)
        outpaths = os.sep + self.clippings.filename + '.png'
        clippings = self.clippings.assign(outpath=outpaths)
        futures: list[Future] = []
        threads = ThreadPoolExecutor()
        CANVAS = self.raster.black.array
        FIG, AX = self.figax
        for itile, clipping in clippings.groupby(level='itile'):
            canvas = CANVAS.copy()
            fig = copy.copy(FIG)
            ax = copy.copy(AX)
            for surface, gdf in clipping.groupby('surface'):
                color = surface.color
                order = surface.order
                gdf.plot(ax=ax, color=color, zorder=order)
