from __future__ import annotations

import numpy as np
from ..inftiles import InferenceTiles
import pandas as pd

from .outtile import GeometryTile
from ..tiles import Tiles, tile

if False:
    from .geotiles import GeometryTiles


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
        instance: GeometryTiles,
        owner,
) -> Padded:
    if instance is None:
        result = self
    elif self.__name__ in instance.attrs:
        result = instance.attrs[self.__name__]
    else:
        inftiles = instance.inftiles
        length = inftiles.outtile.length
        shape = length, length
        boundary = boundary_tiles(shape)
        repeat = len(boundary)

        dx: np.ndarray
        dy: np.ndarray
        dx, dy = boundary.T
        length = instance.inftiles.outtile.length
        xorigin = instance.xtile // length * length
        yorigin = instance.ytile // length * length
        xo = xorigin.values[:, None]
        yo = yorigin.values[:, None]

        pred_xtile = (xo + dx).ravel()
        pred_ytile = (yo + dy).ravel()

        r = np.broadcast_to(dx + 1, xo.shape).ravel()
        c = np.broadcast_to(dy + 1, xo.shape).ravel()

        out_xtile = instance.xtile.repeat(repeat)
        out_ytile = instance.ytile.repeat(repeat)

        arrays = pred_xtile, pred_ytile
        names = 'xtile ytile'.split()
        loc = pd.MultiIndex.from_arrays(arrays, names=names)
        result: Padded = (
            inftiles
            .loc[loc]
            .assign({
                'outtile.xtile': out_xtile,
                'outtile.ytile': out_ytile,
                'outtile.r': r,
                'outtile.c': c,
            })
            .pipe(self.__class__)
        )
        result.attrs.update(inftiles.attrs)
        instance.attrs[self.__name__] = result

    result.geotiles = instance
    return result


class Padded(
    InferenceTiles
):
    geotiles: GeometryTiles = None
    __name__ = 'padded'

    @property
    def inftiles(self):
        return self.geotiles.inftiles

    @Index
    def out(self):
        ...

    @GeometryTile
    def outtile(self):
        ...

