from __future__ import annotations
import copy
import math

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import *

import PIL.Image
import imageio.v3 as iio
import numpy as np
import pandas as pd
import pyproj
import shapely
from PIL import Image

from tile2net.raster import util
from tile2net.tiles import util
from tile2net.tiles.explore import explore
from tile2net.tiles.fixed import GeoDataFrameFixed
from .colormap import ColorMap
from .static import Static
from .tile import Tile
from . import tile

if False:
    import folium
    from .tiles import Tiles
    from ..intiles import InTiles
    from ..segtiles import SegTiles
    from ..vectiles import VecTiles

#
import pandas as pd



class Corners(
    GeoDataFrameFixed
):
    xmin: pd.Series
    ymin: pd.Series
    xmax: pd.Series
    ymax: pd.Series

    @property
    def xtile(self) -> pd.Index:
        """Tile integer X"""
        try:
            return self.index.get_level_values('xtile')
        except KeyError:
            return self['xtile']

    @property
    def ytile(self) -> pd.Index:
        """Tile integer Y"""
        try:
            return self.index.get_level_values('ytile')
        except KeyError:
            return self['ytile']
