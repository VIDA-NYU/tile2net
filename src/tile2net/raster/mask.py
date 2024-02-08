from __future__ import annotations

import functools
import timeit
from tqdm import tqdm
import pyproj
import pickle
import json
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
# from tile2net.raster import Raster
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

if False:
    from tile2net.raster import Raster

class Surface(dict):
    path: str
    usecols: list[str]
    col: Union[int, dict[str, list[str]]]
    color: str
    order: int

    def __getattr__(self, item) -> Surface:
        return self[item]


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

    __getattr__: Callable[[Hashable], Surface]


class Mask:
    """
    clip()
    for cls in clses:
        plot()
    """

    def __init__(
            self,
            raster: Raster,
            geometry: str | Path | GeoDataFrame,
            config: str | Path | dict
    ):
        self.raster = raster
        self.config = config
        self.geometry = geometry

    @property
    def geometry(self):
        return self.__dict__['geometry']

    @geometry.setter
    def geometry(self, value):
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            match value.rsplit('.', 1)[-1]:
                case 'parquet':
                    value = gpd.read_parquet(value)
                case 'feather':
                    value = gpd.read_feather(value)
                case _:
                    value = gpd.read_file(value)
        if not isinstance(value, GeoDataFrame):
            raise ValueError(f'unsupported type: {type(value)}')

        usecols: set[str] = {
            col
            for surface in self.config.values()
            for col in surface.usecols
        }
        gdf = value.loc[:, usecols]

        concat: list[GeoDataFrame] = []
        for name, surface in self.config.items():
            surface: Surface
            loc = np.zeros(len(gdf), dtype=bool)
            if isinstance(surface.col, dict):
                for col, val in surface.col.items():
                    if isinstance(val, list):
                        loc |= gdf[col].isin(val)
                    else:
                        loc |= gdf[col] == val
            else:
                loc = np.ones(len(gdf), dtype=bool)

            concat.append(
                gdf.loc[loc]
                .assign(name=name)
            )

        gdf = pd.concat(concat)
        self.__dict__['geometry'] = gdf

    @property
    def config(self) -> Config:
        return self.__dict__['config']

    @config.setter
    def config(self, value):
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            match value.rsplit('.', 1)[-1]:
                case 'json':
                    with open(value) as f:
                        value = json.load(f)
                case 'pkl':
                    with open(value, 'rb') as f:
                        value = pickle.load(f)
                case _:
                    raise ValueError(f'unsupported file type: {value}')
        if isinstance(value, dict):
            value = Config.from_dict(value)
        else:
            raise ValueError(f'unsupported type: {type(value)}')

        self.__dict__['config'] = value

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

        trans = (
            pyproj.Transformer
            .from_crs(4326, 3857, always_xy=True)
            .transform
        )
        pw, pn = trans(w, n)
        pe, ps = trans(e, s)

        geometry = shapely.box(w, s, e, n)
        filename = Series((
            f'{iy_}_{ix_}_{itile}'
            for iy_, ix_, itile in zip(iy, ix, itiles)
        ), dtype='string')
        result = GeoDataFrame({
            'geometry': geometry,
            'ix': ix,
            'iy': iy,
            'filename': filename,
            'pw': pw,
            'pn': pn,
            'pe': pe,
            'ps': ps,
            'itile': itiles,
        }, crs=4326)
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

    def to_directory(
            self,
            outdir: str | Path = None,
    ) -> Series[str]:

        outdir = os.path.join(outdir, 'annotations')
        os.makedirs(outdir, exist_ok=True)
        clippings = self.clippings
        futures: list[Future] = []
        threads = ThreadPoolExecutor()
        CANVAS = self.raster.black.array
        FIG, AX = self.figax

        total: int = (
            clippings
            .groupby('itile surface'.split())
            .nunique()
            .shape[0]
        )
        outpaths: Series[str] = str(outdir) + os.sep + self.tiles.filename
        tiles = self.tiles.assign(outpath=outpaths)
        TOP = Series.to_dict(tiles.pn)
        LEFT = Series.to_dict(tiles.pw)
        BOTTOM = Series.to_dict(tiles.ps)
        RIGHT = Series.to_dict(tiles.pe)
        OUTPATH = Series.to_dict(tiles.outpath)

        with tqdm(total=total, desc='creating annotation masks') as pbar:
            for itile, clipping in clippings.groupby('itile', sort=False):
                canvas = CANVAS.copy()
                fig = copy.deepcopy(FIG)
                ax = copy.deepcopy(AX)

                outpath = OUTPATH[itile]
                top = TOP[itile]
                left = LEFT[itile]
                bottom = BOTTOM[itile]
                right = RIGHT[itile]

                for surface, tile in clipping.groupby('surface', sort=False):
                    color = self.config.__getattr__(surface).col
                    order = self.config.__getattr__(surface).order
                    tile.plot(
                        ax=ax,
                        color=color,
                        alpha=1,
                        zorder=order,
                        antialiased=False,
                    )
                    ax.imshow(canvas, extent=(top, bottom, right, left))
                    # noinspection PyUnresolvedReferences
                    s, (width, height) = fig.canvas.print_to_buffer()
                    data = np.frombuffer(s, dtype=np.uint8).reshape(width, height, 4)
                    data = data[:, :, 0:3]
                    cv2.imwrite(outpath, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
                    outpath: str
                    array = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                    pbar.update(1)

                # noinspection PyTypeChecker
                future = threads.submit(cv2.imwrite, outpath, array)
                futures.append(future)
                fig.clf()
                plt.close(fig)

        for future in futures:
            future.result()
        threads.shutdown()
        return outpaths


class ClipMask(Mask):
    """
    for cls in clses:
        clip()
        plot()
    """

    @cached_property
    def clippings(self) -> GeoDataFrame:
        tiles = self.tiles
        concat: list[GeoDataFrame] = [
            tiles
            .clip(surface.geometry, keep_geom_type=True)
            .assign(surface=name)
            for name, surface in self.geometry.groupby('surface', sort=False)
        ]
        # noinspection PyTypeChecker
        result: GeoDataFrame = pd.concat(concat)
        return result


class Comparison:
    # noinspection PyUnresolvedReferences
    class Mask(Mask):
        clippings = property(Mask.clippings.func)

    # noinspection PyUnresolvedReferences
    class ClipMask(ClipMask):
        clippings = property(ClipMask.clippings.func)

    def __init__(self, raster, geometry, config):
        self.mask = self.Mask(raster, geometry, config)
        self.clipmask = self.ClipMask(raster, geometry, config)

    def __call__(self, *args, **kwargs):
        mask_time = timeit.timeit(lambda: self.mask.clippings)
        clipmask_time = timeit.timeit(lambda: self.clipmask.clippings)
        print(f"mask.clippings access time: {mask_time:.7f} seconds")
        print(f"clipmask.clippings access time: {clipmask_time:.7f} seconds")


def side_by_side(
        left: Series[str] | str,
        right: Series[str] | str,
) -> Series[str]:
    ...

def annotate(
        self: Raster,
        geometry: str | Path | GeoDataFrame,
        config: str | Path | dict,
        outdir: str | Path = None,
):
    mask = Mask(self, geometry, config)
    result = mask.to_directory(outdir)
    return result





if __name__ == '__main__':
    ...
