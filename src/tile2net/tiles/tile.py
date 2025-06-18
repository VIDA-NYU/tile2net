from __future__ import annotations

import functools
import shutil

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import *

import PIL.Image
import certifi
import geopandas as gpd
import imageio.v3
import imageio.v3 as iio
import numpy as np
import pandas as pd
import pyproj
import requests
import shapely
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm

from tile2net.logger import logger
from tile2net.raster import util
from tile2net.tiles.cfg import cfg
from . import util
from .colormap import ColorMap
from .dir import Dir
from .explore import explore
from .fixed import GeoDataFrameFixed
from .indir import Indir
from .outdir import Outdir
from .source import Source, SourceNotFound

if False:
    import folium
    from .tiles import Tiles



class cached_property:
    __wrapped__ = None

    def __init__(
            self,
            func
    ):
        functools.update_wrapper(self, func)

    def __set_name__(self, owner, name):
        self.__name__ = name
        self.owner = owner

    @functools.cached_property
    def key(self):
        from .tiles import Tiles
        if isinstance(self.owner, Tiles):
            return self.__name__
        elif hasattr(self.owner, 'tiles'):
            return f'{self.owner.tiles.__name__}.{self.__name__}'
        else:
            raise TypeError(
                f'Owner {self.owner} must be a Tiles subclass or have a tiles attribute.'
            )

    def __get__(
            self,
            instance,
            owner
    ):
        if instance is None:
            return self
        key = self.key
        if isinstance(self.owner, Tiles):
            cache = instance.attrs
        elif hasattr(self.owner, 'tiles'):
            cache = instance.tiles.attrs
        else:
            raise TypeError(
                f'Owner {self.owner} must be a Tiles subclass or have a tiles attribute.'
            )
        if key in cache:
            return cache[key]
        result = self.__wrapped__(instance)
        cache[key] = result
        return result

    def __set__(
            self,
            instance,
            value,
    ):
        pass

    def __delete__(
            self,
            instance,
    ):
        pass


def __get__(
        self: Tile,
        instance: Tiles,
        owner: type[Tiles],
) -> Tile:
    self.tiles = instance
    return self

class Tile(

):
    locals().update(
        __get__=__get__
    )
    tiles: Tiles

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    @cached_property
    def scale(self):
        """
        Tile scale; the XYZ scale of the tiles.
        Higher value means smaller area.
        """
        return self.zoom

    @cached_property
    def zoom(self) -> int:
        """Tile zoom level"""
        result = self.tiles.cfg.zoom
        if not result:
            msg = 'Zoom not set in configuration'
            raise ValueError(msg)
        return result

    @cached_property
    def dimension(self):
        """Tile dimension; inferred from input files"""
        try:
            sample = next(p for p in self.tiles.file if Path(p).is_file())
        except StopIteration:
            raise FileNotFoundError('No image files found to infer dimension.')
        return iio.imread(sample).shape[1]  # width
