from __future__ import annotations
from numpy import ndarray
import rasterio

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

import numpy as np
import pandas as pd

from .tiles import Tiles
from .outdir import Outdir


class Stitched(
    Tiles
):

    @property
    def tiles(self) -> Tiles:
        """Tiles object that this Stitched object is based on"""
        try:
            return self.attrs['tiles']
        except KeyError as e:
            raise AttributeError(

            ) from e

    @tiles.setter
    def tiles(self, value: Tiles):
        """Set the Tiles object that this Stitched object is based on"""
        if not isinstance(value, Tiles):
            raise TypeError(f"Expected Tiles object, got {type(value)}")
        self.attrs['tiles'] = value

    @property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        if 'dimension' in self.attrs:
            return self.attrs['dimension']

        tiles = self.tiles
        dscale = int(self.tscale - tiles.tscale)
        mlength = dscale ** 2
        result = tiles.dimension * mlength
        self.attrs['dimension'] = result
        return result

    @property
    def mlength(self):
        tiles = self.tiles
        dscale = int(self.tscale - tiles.tscale)
        return dscale ** 2

    @property
    def group(self) -> pd.Series:
        if 'group' in self.columns:
            return self['group']
        result = pd.Series(np.arange(len(self)), index=self.index)
        self['group'] = result
        return self['group']

    @property
    def outdir(self):
        # return self.tiles.outdir
        tiles = self.tiles
        tiles._stitched = self
        return tiles.outdir

    @property
    def affine_params(self) -> pd.Series:
        key = 'affine_params'
        if key in self:
            return self[key]
        it = zip(self.gw, self.gs, self.ge, self.gn)
        dim = self.dimension
        data = [
            rasterio.transform
            .from_bounds(gw, gs, ge, gn, dim, dim)
            for gw, gs, ge, gn in it
        ]
        result = pd.Series(data, index=self.index, name=key)
        self[key] = result
        return self[key]

    def affine_iterator(self, *args, **kwargs) -> Iterator[ndarray]:
        key = self._trace
        cache = self.tiles.attrs
        if key in cache:
            it = cache[key]
        else:
            affine = self.affine_params

            def gen():
                n = cfg.model.bs_val
                a = affine.to_numpy()
                q, r = divmod(len(a), n)
                yield from a[:q * n].reshape(q, n)
                if r:
                    yield a[-r:]

            it = gen()
            cache[key] = it
        yield from it
        del cache[key]
