from __future__ import annotations

import hashlib
import math
import os
import tempfile
from functools import *
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import shapely
import tqdm
from PIL import ImageColor, Image
from geopandas import GeoDataFrame
from pandas import MultiIndex, Series, Index

from tile2net.grid import frame, util
from tile2net.grid.basegrid.file import File
from tile2net.grid.cfg import cfg, Cfg
from tile2net.grid.cfg.logger import logger
from tile2net.grid.explore import explore
from tile2net.grid.frame.framewrapper import FrameWrapper
from tile2net.grid.loaders.dataloader import BaseDataLoader
from tile2net.grid.loaders.datawrapper import DataWrapper
from tile2net.grid.loaders.rescale import RescaleDataSet
from tile2net.grid.loaders.stitch import StitchWriterDataSet
from tile2net.grid.loaders.unstitch import UnstitchDataSet, UnstitchDataWrapper
from tile2net.grid.sampler.benchmark import Benchmark

if TYPE_CHECKING:
    import folium
    from tile2net.grid.seggrid.seggrid import SegGrid
    from tile2net.grid.grid import Grid
    from tile2net.grid.vecgrid.vecgrid import VecGrid
    from tile2net.grid.dir.outdir import Outdir


class BaseGrid(
    FrameWrapper,
):


    @frame.column
    def itile(self):
        """
        Simple sequential integer identifier for each tile in the grid.

        Example:
            >>> grid: Grid
            >>> grid.itile
        xtile   ytile
        317280  387840       0
                          ...
                387871    1023
        """
        return np.arange(len(self))

    @cached_property
    def hash(self) -> str:
        """Hash of the Tiles in the grid and the configuration."""
        pairs = (
            self.index
            .to_frame(index=False)  # -> DataFrame with ['xtile', 'ytile']
            .astype({'xtile': 'int64', 'ytile': 'int64'}, copy=False)
            .to_numpy(copy=False)  # -> (n, 2) int64 ndarray
        )
        tiles = hashlib.blake2b(
            np.ascontiguousarray(pairs).tobytes(),
            digest_size=8,
        ).hexdigest()
        cfg = self.cfg.hash()
        result = f'{tiles}-{cfg}'
        return result


    scale: int
    """Tile scale; the XYZ scale of the grid.
    Higher value means smaller area.
    """

    @property
    def dimension(self):
        """
        Pixel dimension of each tile

        Computed as grid.dimension * self.length. For example, if Grid tiles
        are 256x256 pixels and length is 4, tiles are 1024x1024 pixels.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.dimension
            1024
        """
        result = self.grid.dimension * self.length
        return result

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of a tile in pixels"""
        return self.dimension, self.dimension

    @File
    def file(self):
        """
        Namespace for file attributes

        Example:
            >>> self.file.Static
            xtile   ytile
            317280  387840    /home/<user>/tile2net/ma/grid/static/20/31...
                    387841    /home/<user>/tile2net/ma/grid/static/20/31...
        """

    @property
    def grid(self) -> Grid:
        """Reference to the Grid instance"""
        return self.instance

    @property
    def seggrid(self) -> SegGrid:
        """Reference to the SegGrid instance"""
        return self.grid.seggrid

    @property
    def vecgrid(self) -> VecGrid:
        """Reference to the VecGrid instance"""
        return self.grid.vecgrid

    location: str = None
    """Location passed by the user when instantiating the Grid"""


    @Cfg
    def cfg(self):
        """
        Namespace container for configuration options of a Grid.

        Example:
            >>> grid: BaseGrid
            # access zoom level config
            >>> grid.cfg.zoom
            20
            # access batch size config
            >>> grid.cfg.validation.batch_size
            32
            # access polygon param
            >>> grid.cfg.polygon.max_hole_area
            1000
        """

    @property
    def index(self) -> MultiIndex:
        return self.frame.index

    @property
    def outdir(self) -> Outdir:
        """
        Output in which the results, such as annotated images and geometry, will be stored:
        Example:
            >>> grid: Grid
            >>> grid.outdir
            Outdir(
                format='/home/<user>/tile2net/{z}/{x}_{y}',
                dir='/home/<user>/tile2net',
                original='/home/<user>/tile2net/z/x_y',
                suffix='z/x_y'
            )

        Setting the output directory:
        >>> grid: Grid
        >>> grid = grid.set_outdir('/path/to/output')
        """

        return self.grid.outdir

    @property
    def tempdir(self):
        """
        Temporary directory for intermediate processing files.

        Example:
            >>> grid: Grid
            >>> grid.tempdir
            Tempdir(
                dir='/tmp/tile2net/ma/grid/static'
                original='/tmp/tile2net/ma/grid/static/z/x_y',
            )
        """
        return self.grid.tempdir

    def __len__(self):
        return len(self.frame)

    def __repr__(self):
        result = f'{self.__class__.__qualname__}:\n\n'
        try:
            result += (
                f'Source: \n\t'
                f'{self.grid.source}\n'
            )
        except Exception:
            ...
        result += f'\n'
        result += self.frame.__repr__()
        return result

    def pipe(self, *args, **kwargs):
        func = args[0] if args else kwargs.pop('func', None)
        if func is None:
            raise ValueError('func must be provided to pipe')
        result = func(self, *args[1:], **kwargs)
        return result

    @property
    def crs(self):
        return self.crs

    def __delete__(
            self,
            instance: BaseGrid,
    ):
        try:
            del instance.frame.__dict__[self.__name__]
        except KeyError:
            ...

    def __set__(
            self,
            instance: BaseGrid,
            value,
    ):
        # todo: maybe should be copy
        instance.frame.__dict__[self.__name__] = value

    @cached_property
    def colormap(self):
        """
        Callable which applies colormaps to tensors, ndarrays, and images.

        Update cfg.label2color to change the colormap:
            >>> self.cfg.label2color
            {'sidewalk': 'red',
             'road': 'cyan',
             'crosswalk': 'yellow',
             'curb': 'blue',
             'void': 'black'}

        Example:
            >>> self.colormap
            ColorMap(
              0 -> [255, 0, 0]   (sidewalk -> red)
              1 -> [0, 255, 255] (road -> cyan)
              2 -> [255, 255, 0] (crosswalk -> yellow)
              3 -> [0, 0, 0]     (void -> black)
            )

            >>> self.colormap(np.array([[0,1,2]]))
            Out[11]:
            array([[[  0,   0, 255],
                    [  0, 128,   0],
                    [255,   0,   0]]], dtype=uint8)
        """
        return self.cfg.colormap

    @cached_property
    def time_usage(self) -> float:
        return 0

    @cached_property
    def disk_usage(self) -> int:
        return 0

    @cached_property
    def sampler(self) -> Benchmark:
        result = Benchmark(include_gpu=True)
        return result

