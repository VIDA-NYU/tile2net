from __future__ import annotations

from distutils.command.install_egg_info import install_egg_info

import numpy as np
import pandas as pd

from . import vectile
from .segtiles import SegTiles
from ..tiles import tile

if False:
    from tile2net.tiles.vectiles.vectiles import VecTiles


class VecTile(
    vectile.VecTile
):

    @property
    def r(self) -> pd.Series:
        """row within the segtile of this tile"""

    @property
    def c(self) -> pd.Series:
        """column within the segtile of this tile"""


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
        instance: VecTiles,
        owner,
) -> Padded:
    if instance is None:
        result = self
    elif self.__name__ in instance.attrs:
        result = instance.attrs[self.__name__]
    else:
        segtiles = instance.segtiles
        vectiles = segtiles.vectiles
        length = vectiles.tile.length // segtiles.tile.length

        scale = segtiles.tile.scale
        corners = vectiles.corners(scale)
        xmin = corners.xmin - 1
        ymin = corners.ymin - 1
        xmax = corners.xmax + 1
        ymax = corners.ymax + 1
        padded = segtiles.from_ranges(xmin, ymin, xmax, ymax, scale=scale)
        area = length ** 2
        assert len(padded) == len(segtiles) * area
        vectile_xtile = vectiles.xtile.repeat(area)
        vectile_ytile = vectiles.ytile.repeat(area)


        shape = length, length
        boundary = boundary_tiles(shape)
        repeat = len(boundary)

        vectiles.to_scale(segtiles.tile.scale)

        dx: np.ndarray
        dy: np.ndarray
        dx, dy = boundary.T
        length = vectiles.segtiles.vectile.length
        xorigin = vectiles.xtile // length * length
        yorigin = vectiles.ytile // length * length
        xo = xorigin.values[:, None]
        yo = yorigin.values[:, None]

        pred_xtile = (xo + dx).ravel()
        pred_ytile = (yo + dy).ravel()
        r = np.broadcast_to(dx + 1, xo.shape).ravel()
        c = np.broadcast_to(dy + 1, xo.shape).ravel()

        out_xtile = vectiles.xtile.repeat(repeat)
        out_ytile = vectiles.ytile.repeat(repeat)

        arrays = pred_xtile, pred_ytile
        names = 'xtile ytile'.split()
        loc = pd.MultiIndex.from_arrays(arrays, names=names)
        result: Padded = (
            segtiles
            .loc[loc]
            .assign({
                'vectile.xtile': out_xtile,
                'vectile.ytile': out_ytile,
                'vectile.r': r,
                'vectile.c': c,
            })
            .pipe(self.__class__)
        )
        result.attrs.update(segtiles.attrs)
        vectiles.attrs[self.__name__] = result

    result.vectiles = instance
    return result


class Padded(
    SegTiles,
):
    __name__ = 'padded'
    locals().update(
        __get__=__get__,
    )

    @tile.cached_property
    def segtiles(self) -> SegTiles:
        ...

    @Index
    def out(self):
        ...

    @VecTile
    def vectile(self):
        ...
