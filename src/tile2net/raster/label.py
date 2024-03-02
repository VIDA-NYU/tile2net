from __future__ import annotations

import itertools
import json
import os.path
import pickle
import tempfile
from collections import *
from concurrent.futures import *
from functools import *
from pathlib import Path
from typing import *

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import shapely
from geopandas import GeoDataFrame
from numpy import ndarray
from pandas import Series, Index
from tqdm import tqdm

from tile2net.raster.tile import Tile

if False:
    from tile2net.raster import Raster


class Surface(dict):
    path: str | GeoDataFrame
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

    def __getattr__(self, item) -> Surface:
        return self[item]

    __getitem__: Callable[[Hashable], Surface]


class Mask:

    def __init__(self, raster: Raster, config: str | Path | dict):
        self.raster = raster
        self.config = config

    @cached_property
    def geometry(self) -> GeoDataFrame:
        """
        In the config, 'path' may be str, collection of strs, or GeoDataFrame;
        here we concat the various sources into a single GeoDataFrame
        """
        threads = ThreadPoolExecutor()
        surface_paths: dict[str, set[str]] = defaultdict(set)
        surface_frame: dict[str, GeoDataFrame] = {}
        path_surfaces: dict[str, set[str]] = defaultdict(set)
        config = self.config

        # paths might be shared between surfaces; avoid loading duplicates
        # iterate across paths and map which frames each surface uses
        for name, surface in self.config.items():
            path = surface.path

            if isinstance(path, Path):
                path = str(path)

            if isinstance(path, str):
                surface_paths[name].add(path)

            elif isinstance(path, (list, tuple, set)):
                for p in path:
                    surface_paths[name].add(p)

            elif isinstance(path, GeoDataFrame):
                surface_frame[name] = path

            else:
                raise ValueError(f'unsupported type: {type(path)}')

        # reverse the mapping so that instead of many-to-one it's one-to-many
        for surface, paths in surface_paths.items():
            for path in paths:
                path_surfaces[path].add(surface)

        # noinspection PyTypeChecker
        def read(path: str) -> Future:
            match path.rsplit('.', 1)[-1]:
                case 'parquet':
                    return threads.submit(gpd.read_parquet, path)
                case 'feather':
                    return threads.submit(gpd.read_feather, path)
                case _:
                    return threads.submit(gpd.read_file, path)

        paths: set[str] = set(itertools.chain.from_iterable(surface_paths.values()))

        futures = {path: read(path) for path in paths}
        paths_frames: Iterator[tuple[str, GeoDataFrame]] = (
            (path, future.result())
            for path, future in futures.items()
        )
        paths_frames = list(paths_frames)

        # generate a mapping of each surface to the frame it uses
        surface_frame.update({
            surface: frame
            for path, frame in paths_frames
            for surface in path_surfaces[path]
        })
        id_frame = {
            id(frame): frame.to_crs(3857)
            for frame in surface_frame.values()
        }
        surface_frame = {
            surface: id_frame[id(frame)]
            for surface, frame in surface_frame.items()
        }

        # generate a list of frames to concatenate with the surface name
        concat: list[GeoDataFrame] = []
        for name, frame in surface_frame.items():
            frame: GeoDataFrame
            usecols = config[name].usecols
            if not isinstance(usecols, (list, set, tuple)):
                usecols = slice(None)
            loc = np.zeros(len(frame), dtype=bool)
            surface = config[name]

            # if user specified cols, mask where the cols are equal to the specification
            if isinstance(surface.col, dict):
                for col, val in surface.col.items():
                    if isinstance(val, list):
                        loc |= frame[col].isin(val)
                    else:
                        loc |= frame[col] == val
            else:
                loc = np.ones(len(frame), dtype=bool)

            append = frame.loc[loc, usecols].assign(surface=name)
            concat.append(append)

        # noinspection PyTypeChecker
        result: GeoDataFrame = pd.concat(concat)
        return result

    @property
    def config(self) -> Config:
        return self.__dict__['config']

    @config.setter
    def config(self, value):
        # allows config to be a path to a .json file
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
        # tiles: list[Tile] = self.raster.tiles.ravel().tolist()
        step = self.raster.stitch_step
        tiles: list[Tile] = (
            self.raster.tiles
            # [::step, ::step]
            .ravel()
            .tolist()
        )
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

        gn = fromiter('top')
        gs = fromiter('bottom')
        gw = fromiter('left')
        ge = fromiter('right')
        iy = fromiter('ytile', dtype=int)
        ix = fromiter('xtile', dtype=int)
        itiles = fromiter('idd', dtype=int)

        trans = (
            pyproj.Transformer
            .from_crs(4326, 3857, always_xy=True)
            .transform
        )
        pw, pn = trans(gw, gn)
        pe, ps = trans(ge, gs)

        # vectorized some data about the tiles to be used
        geometry = shapely.box(pw, ps, pe, pn)
        filename = Series((
            f'{iy_}_{ix_}_{itile}.png'
            for iy_, ix_, itile in zip(iy, ix, itiles)
        ), dtype='string')
        index = Index(itiles, name='itile')
        result = GeoDataFrame({
            'geometry': geometry,
            'ix': ix,
            'iy': iy,
            'filename': filename,
            'pw': pw,
            'pn': pn,
            'pe': pe,
            'ps': ps,
            'gn': gn,
            'gs': gs,
            'gw': gw,
            'ge': ge,
            'itile': itiles,
        }, crs=3857, index=index)
        return result

    @cached_property
    def tiles(self) -> GeoDataFrame:
        raster = self.raster
        step = raster.stitch_step
        indices = np.arange(raster.tiles.size).reshape((raster.width, raster.height))
        indices: ndarray = (
            indices
            # iterate by step to get the top left tile of each new merged tile
            # [:r:step, :c:step]
            [::step, ::step]
            # reshape to broadcast so offsets can be added
            .reshape((-1, 1, 1))
            # add offsets to get the indices of the tiles to merge
            .__add__(indices[:step, :step])
            # flatten to get a list of merged tiles
            .reshape((-1, step * step))
        )
        count = len(indices)
        tiles = raster.tiles.ravel().tolist()

        def fromiter(attr: str, item: int, dtype=float) -> ndarray:
            I = (
                square[item]
                for square in indices
            )
            return np.fromiter((
                tiles[i].__getattribute__(attr)
                for i in I
            ), dtype=dtype, count=count)

        gn = fromiter('top', 0)
        gs = fromiter('bottom', -1)
        gw = fromiter('left', 0)
        ge = fromiter('right', -1)

        trans = (
            pyproj.Transformer
            .from_crs(4326, 3857, always_xy=True)
            .transform
        )
        pw, pn = trans(gw, gn)
        pe, ps = trans(ge, gs)

        # vectorized some data about the tiles to be used
        geometry = shapely.box(pw, ps, pe, pn)
        # index = Index(itiles, name='itile')
        filename = self.raster.project.tiles.annotated.files(
            raster.tiles[::step, ::step],
        )
        filename = list(filename)
        itiles = np.arange(len(filename))
        index = Index(itiles, name='itile')

        result = GeoDataFrame({
            'geometry': geometry,
            'filename': filename,
            'pw': pw,
            'pn': pn,
            'pe': pe,
            'ps': ps,
            'gn': gn,
            'gs': gs,
            'gw': gw,
            'ge': ge,
            'itile': itiles,
        }, crs=3857, index=index)
        return result

    @cached_property
    def clippings(self) -> GeoDataFrame:
        # clip the geometry to the tiles
        matches: GeoDataFrame = (
            self
            .geometry
            .sjoin(self.tiles, how='inner', op='intersects')
        )
        tiles = self.tiles.loc[matches.itile]
        matches.geometry = matches.intersection(tiles.geometry, align=False)
        return matches

    def to_directory(self, outdir: str | Path = None, ) -> Series[str]:
        """
        For each tile, create a mask of the ground truth data and
        save it to a directory. If outdir is None, tempdir is used.
        """
        if outdir is None:
            outdir = tempfile.mkdtemp()
        outdir = os.path.join(outdir, 'annotations')
        os.makedirs(outdir, exist_ok=True)
        clippings = self.clippings
        threads = ThreadPoolExecutor()

        # there is one iteration for each tile for each surface
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

        def save(fig: plt.Figure, outpath: str):
            fig.savefig(
                outpath,
                dpi=1200,
                facecolor='black',
                bbox_inches='tight',
                pad_inches=0,
            )
            fig.clf()
            plt.close(fig)

        with tqdm(total=total, desc='creating annotation masks') as pbar:
            futures = []
            img = self.raster.black.array
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dpi = 1200
            figsize = (img.shape[1] / dpi, img.shape[0] / dpi)

            for itile, clipping in clippings.groupby('itile', sort=False):
                # determine bounds and outpath
                outpath = OUTPATH[itile]
                top = TOP[itile]
                left = LEFT[itile]
                bottom = BOTTOM[itile]
                right = RIGHT[itile]

                # configure the figure--I tried to create
                # copies of fig & ax but this caused errors
                # one figure per tile
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                plt.box(False)
                fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                ax.axis('off')
                ax.set_xlim(left, right)
                ax.set_ylim(bottom, top)
                ax.set_facecolor('black')

                # for each surface in the tile, plot
                for surface, tile in clipping.groupby('surface', sort=False):
                    tile: GeoDataFrame
                    color = self.config[surface].color
                    zorder = self.config[surface].order
                    pbar.update(1)
                    tile.plot(ax=ax, color=color, antialiased=False, zorder=zorder, alpha=1)

                # noinspection PyTypeChecker
                future = threads.submit(save, fig, outpath)
                futures.append(future)

        for future in futures:
            future.result()
        threads.shutdown()
        return outpaths


def label(
        self: Raster,
        config: str | Path | dict,
        outdir: str | Path = None,
        stitch_step=None,
) -> Series[str]:
    """
    "Annotate," "mask," or "label" the tiles with ground truth data;
    generate new images from external sources to train the model.

    Parameters
    ----------
    self : Raster
        The raster which defines the tiles to be labeled
    config : str | dict
        a dict or path to .json containing the configuration metadata, for example:
        ```
        config = {
            'road': {
                'path': rd,
                'usecols': ['geometry', 'TYPE'],
                'col': {'TYPE': ['RD-ALLEY', 'RD-UNPAVED', 'RD-PAVED']},
                'color': 'green',
                'order': 16
            }
        }
        ```
    outdir : str | Path
        the directory to save the labeled images to
    stitch_step : int
        if not None, the step to use when stitching tiles together

    Returns
    -------
    Series[str]
        a series of paths to the labeled images, where the index
        is the idd of the tile
    """
    if stitch_step is not None:
        self.update(stitch_step)
    mask = Mask(self, config)
    result = mask.to_directory(outdir)
    return result


if __name__ == '__main__':
    from tile2net.raster import Raster

    raster = Raster(
        # location='Cambridge, MA',
        location='42.35725124845672, -71.09518965878434, 42.36809181753787, -71.07791143338436',
        zoom=18,
    )
    raster.update(2)
    result = label(
        raster,
        config={
            'road': {
                'path': 'https://gis.cambridgema.gov/download/gdb/BASEMAP_Driveways.gdb.zip',
                'usecols': ['geometry', 'TYPE'],
                'col': {'TYPE': ['DRIVE-UNP', 'DRIVE-PAV', 'RD-PAVED']},
                'color': 'green',
                'order': 16
            }
        }
    )
    result
