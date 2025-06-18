from __future__ import annotations
from ..fixed import GeoDataFrameFixed
import pandas as pd

from typing import *

import numpy as np
from ..predtiles import PredTiles
from .. import predtiles
import pandas as pd
import rasterio

from .mosaic import Mosaic
from .. import tile
from ..tiles import Tiles

if False:
    from .outtiles import OutTiles


def boundary_tiles(
        shape: tuple[int, int],
        pad: int = 1,
) -> np.ndarray:
    rows, cols = shape
    r = np.arange(-pad, rows + pad)
    c = np.arange(-pad, cols + pad)
    R, C = np.meshgrid(r, c, indexing='ij')
    coords = np.column_stack((R.ravel(), C.ravel()))
    mask = coords[:, 0] < 0
    mask |= coords[:, 0] >= rows
    mask |= coords[:, 1] < 0
    mask |= coords[:, 1] >= cols
    return coords[mask]


def __get__(
        self: Index,
        instance: Padded,
        owner
):
    self.padded = instance
    return self


class Index(

):
    padded: Padded = None
    locals().update(
        __get__=__get__
    )

    def __init__(self, *args, ):
        ...

    def __set_name__(self, owner, name):
        self.__name__ = name

    @property
    def xtile(self) -> pd.Series:
        return self.padded[f'{self.__name__}.xtile']

    @property
    def ytile(self) -> pd.Series:
        return self.padded[f'{self.__name__}.ytile']

    @tile.cached_property
    def index(self) -> pd.MultiIndex:
        xtile = self.xtile
        ytile = self.ytile
        arrays = xtile, ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result


def __get__(
        self: Padded,
        instance: OutTiles,
        owner,
) -> Padded:
    if instance is None:
        result = self
    elif self.__name__ in instance.attrs:
        result = instance.attrs[self.__name__]
    else:
        predtiles = instance.predtiles
        length = predtiles.mosaic.length
        shape = length, length
        boundary = boundary_tiles(shape)
        repeat = len(boundary)

        dx: np.ndarray
        dy: np.ndarray
        dx, dy = boundary.T
        xo = instance.xorigin.values[:, None]
        yo = instance.yorigin.values[:, None]

        pred_xtile = (xo + dx).ravel()
        pred_ytile = (yo + dy).ravel()

        r = np.broadcast_to(dx + 1, xo.shape).ravel()
        c = np.broadcast_to(dy + 1, xo.shape).ravel()

        out_xtile = instance.xtile.repeat(repeat)
        out_ytile = instance.ytile.repeat(repeat)

        arrays = pred_xtile, pred_ytile
        names = 'xtile ytile'.split()
        loc = pd.MultiIndex.from_arrays(arrays, names=names)
        result = (
            predtiles
            .loc[loc]
            .assign({
                'out.xtile': out_xtile,
                'out.ytile': out_ytile,
                'mosaic.r': r,
                'mosaic.c': c,
            })
            .pipe(Padded)
        )
        result.attrs.update(predtiles.attrs)
        instance.attrs[self.__name__] = result

    result.outtiles = instance
    return result


class Padded(
    PredTiles
):
    outtiles: OutTiles = None
    __name__ = 'padded'

    @Index
    def out(self):
        ...

    @Mosaic
    def mosaic(self):
        ...
