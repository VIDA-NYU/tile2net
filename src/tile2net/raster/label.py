from __future__ import annotations

import requests
from urllib.parse import urlparse
import tempfile

import copy
import itertools
import json
import os.path
import pickle
import timeit
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
    import folium


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
    """
    clip()
    for cls in clses:
        plot()
    """

    def __init__(self, raster: Raster, config: str | Path | dict):
        self.raster = raster
        self.config = config

    @staticmethod
    def download(path: str, url: str) -> str:
        if os.path.exists(path):
            return path

        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        return path

    @cached_property
    def url_paths(self) -> dict[str, str]:
        threads = ThreadPoolExecutor()
        directory = tempfile.mkdtemp()
        url_paths: dict[str, str] = {}
        files: set[str] = set()
        # todo: we have to accept lists of urls
        for name, surface in self.config.items():
            url = surface.path
            if isinstance(url, Path):
                url = str(url)
            if not isinstance(url, str):
                continue
            if urlparse(url).scheme in ('http', 'https'):
                if url in url_paths:
                    continue
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                filepath = os.path.join(directory, filename)
                url_paths[url] = filepath
                files.add(filepath)

        paths_urls: dict[str, str] = {
            path: url
            for url, path in url_paths.items()
        }

        for _ in threads.map(self.download, paths_urls.keys(), paths_urls.values()):
            ...

        return url_paths

    @cached_property
    def url_paths(self) -> dict[str, str]:
        """ download missing files and map urls to paths """
        threads = ThreadPoolExecutor()
        directory = tempfile.mkdtemp()
        url_paths: dict[str, str] = {}

        for name, surface in self.config.items():
            url = surface.path
            if isinstance(url, str):
                urls = [url]
            elif isinstance(url, (list, set, tuple)):
                urls = url
            else:
                continue
            for url in urls:
                if urlparse(url).scheme in ('http', 'https'):
                    if url in url_paths:
                        continue
                    parsed_url = urlparse(url)
                    filename = os.path.basename(parsed_url.path)
                    path = os.path.join(directory, filename)
                    url_paths[url] = path

        paths_urls: dict[str, str] = {
            path: url
            for url, path in url_paths.items()
        }
        futures = [
            threads.submit(self.download, path, url)
            for path, url in paths_urls.items()
        ]
        for future in futures:
            future.result()

        return url_paths

    @cached_property
    def geometry(self) -> GeoDataFrame:
        threads = ThreadPoolExecutor()
        surface_paths: dict[str, set[str]] = defaultdict(set)
        surface_frame: dict[str, GeoDataFrame] = {}
        path_surfaces: dict[str, set[str]] = defaultdict(set)
        config = self.config

        url_paths = self.url_paths

        for name, surface in self.config.items():
            path = surface.path

            if isinstance(path, Path):
                path = str(path)

            if isinstance(path, str):
                if urlparse(path).scheme in ('http', 'https'):
                    path = url_paths[path]
                surface_paths[name].add(path)

            elif isinstance(path, (list, tuple, set)):
                for p in path:
                    if urlparse(p).scheme in ('http', 'https'):
                        p = url_paths[p]
                    surface_paths[name].add(p)

            elif isinstance(path, GeoDataFrame):
                surface_frame[name] = path

            else:
                raise ValueError(f'unsupported type: {type(path)}')

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

        concat: list[GeoDataFrame] = []
        for name, frame in surface_frame.items():
            frame: GeoDataFrame
            usecols = config[name].usecols
            if not isinstance(usecols, (list, set, tuple)):
                usecols = slice(None)
            loc = np.zeros(len(frame), dtype=bool)
            surface = config[name]
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
        # tiles: list[Tile] = self.raster.tiles.tolist()
        tiles: list[Tile] = self.raster.tiles.ravel().tolist()
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

        # geometry = shapely.box(gw, gs, ge, gn)
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
        # }, crs=4326)
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

    def to_directory(self, outdir: str | Path = None, ) -> Series[str]:
        if outdir is None:
            outdir = tempfile.mkdtemp()
        outdir = os.path.join(outdir, 'annotations')
        os.makedirs(outdir, exist_ok=True)
        clippings = self.clippings
        threads = ThreadPoolExecutor()

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
                outpath = OUTPATH[itile]
                top = TOP[itile]
                left = LEFT[itile]
                bottom = BOTTOM[itile]
                right = RIGHT[itile]

                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                plt.box(False)
                fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                ax.axis('off')
                ax.set_xlim(left, right)
                ax.set_ylim(bottom, top)
                ax.set_facecolor('black')

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

    @classmethod
    def from_example(cls):
        from tile2net.raster.raster import Raster
        raster = Raster(
            # location='Cambridge, MA',
            location='42.35725124845672, -71.09518965878434, 42.36809181753787, -71.07791143338436',
            zoom=18,
        )
        rd = '/home/arstneio/Downloads/BASEMAP_Roads (1).geojson'
        pl = '/home/arstneio/Downloads/BASEMAP_ParkingLots.shp/BASEMAP_ParkingLots.shp'
        config = {
            'parkinglot': {
                'path': pl,
                'usecols': ['geometry'],
                'col': -1,
                'color': 'grey',
                'order': 10
            },
            'road': {
                'path': rd,
                'usecols': ['geometry', 'TYPE'],
                'col': {'TYPE': ['RD-ALLEY', 'RD-UNPAVED', 'RD-PAVED']},
                'color': 'green',
                'order': 16
            },
            'median': {
                'path': rd,
                'usecols': ['geometry', 'TYPE'],
                'col': {'TYPE': 'RD-TRAF-ISLAND'},
                'color': 'black',
                'order': 28
            },
            'alley': {
                'path': rd,
                'usecols': ['geometry', 'TYPE'],
                'col': {'TYPE': 'RD-ALLEY'},
                'color': 'bisque',
                'order': 12
            }
        }
        result = cls(raster, config)
        return result


class ClipMask(Mask):
    # clipmask is slower
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
            .clip(surface.geometry, keep_geom_type=False)
            .assign(surface=name)
            for name, surface in self.geometry.groupby('surface', sort=False)
        ]
        # noinspection PyTypeChecker
        result: GeoDataFrame = pd.concat(concat).reset_index(drop=True)
        return result


class Comparison:
    # todo: compare both runtime and appearance

    # noinspection PyUnresolvedReferences
    class Mask(Mask):
        clippings = property(Mask.clippings.func)

    # noinspection PyUnresolvedReferences
    class ClipMask(ClipMask):
        clippings = property(ClipMask.clippings.func)

    def __init__(self, raster, config):
        self.mask = self.Mask(raster, config)
        self.clipmask = self.ClipMask(raster, config)
        # save time by reusing the same geometry
        self.config = self.clipmask.config = self.mask.config
        self.clipmask.geometry = self.mask.geometry
        # self.config = config

    def runtime(self):
        mask_time = timeit.timeit(lambda: self.mask.clippings)
        clipmask_time = timeit.timeit(lambda: self.clipmask.clippings)
        print(f"mask.clippings access time: {mask_time:.7f} seconds")
        print(f"clipmask.clippings access time: {clipmask_time:.7f} seconds")

    @cached_property
    def MASK(self) -> GeoDataFrame:
        return self.mask.clippings

    @cached_property
    def CLIPMASK(self) -> GeoDataFrame:
        return self.clipmask.clippings

    def explore(self, name=None, *args, **kwargs) -> folium.Map:
        """
        for each gdf, for each surface, plot geometry with color

        """
        import folium
        kwargs.setdefault('tiles', 'cartodbdark_matter')
        kwargs.setdefault('style_kwds', dict(weight=5, radius=5))
        MASK = self.MASK['geometry surface itile'.split()].assign(source='mask')
        CLIPMASK = self.CLIPMASK['geometry surface itile'.split()].assign(source='clipmask')
        # noinspection PyTypeChecker
        m: folium.Map = None
        colors = iter('red yellow green orange blue purple brown pink'.split())
        for name, surface in self.config.items():
            surface: Surface
            mask: GeoDataFrame = MASK.loc[MASK.surface == name]
            clipmask: GeoDataFrame = CLIPMASK.loc[CLIPMASK.surface == name]
            color = next(colors)
            if len(mask):
                m = mask.explore(
                    color=color,
                    *args,
                    **kwargs,
                    m=m,
                    name=f'{name} Mask',
                    show=False,
                    overlay=True,
                )
            if len(clipmask):
                m = clipmask.explore(
                    color=color,
                    *args,
                    **kwargs,
                    m=m,
                    name=f'{name} ClipMask',
                    show=False,
                    overlay=True,
                )
        folium.LayerControl().add_to(m)

        return m

    @classmethod
    def from_example(cls):
        from tile2net.raster.raster import Raster
        raster = Raster(
            # location='Cambridge, MA',
            location='42.35725124845672, -71.09518965878434, 42.36809181753787, -71.07791143338436',
            zoom=18,
        )
        rd = '/home/arstneio/Downloads/BASEMAP_Roads (1).geojson'
        pl = '/home/arstneio/Downloads/BASEMAP_ParkingLots.shp/BASEMAP_ParkingLots.shp'
        config = {
            'parkinglot': {
                'path': pl,
                'usecols': ['geometry'],
                'col': -1,
                'color': 'grey',
                'order': 10
            },
            'road': {
                'path': rd,
                'usecols': ['geometry', 'TYPE'],
                'col': {'TYPE': ['RD-ALLEY', 'RD-UNPAVED', 'RD-PAVED']},
                'color': 'green',
                'order': 16
            },
            'median': {
                'path': rd,
                'usecols': ['geometry', 'TYPE'],
                'col': {'TYPE': 'RD-TRAF-ISLAND'},
                'color': 'black',
                'order': 28
            },
            'alley': {
                'path': rd,
                'usecols': ['geometry', 'TYPE'],
                'col': {'TYPE': 'RD-ALLEY'},
                'color': 'bisque',
                'order': 12
            }
        }
        comparison = cls(raster, config)
        return comparison



def label(
        self: Raster,
        config: str | Path | dict,
        outdir: str | Path = None,
) -> Series[str]:
    """
    "Annotate", "mask," or "label" the tiles with ground truth data;
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

    Returns
    -------
    Series[str]
        a series of paths to the labeled images, where the index
        is the idd of the tile
    """
    mask = Mask(self, config)
    result = mask.to_directory(outdir)
    return result


if __name__ == '__main__':
    files = Mask.from_example().to_directory()
    files
