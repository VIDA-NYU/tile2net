from __future__ import annotations

import numpy as np
import pandas as pd

from . import vectile
from .segtiles import SegTiles
from ..tiles import tile


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
        instance: Broadcast,
        owner
):
    self.padded = instance
    return self


class Index(

):
    padded: Broadcast = None
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
        self: Broadcast,
        instance: SegTiles,
        owner,
) -> Broadcast:
    if instance is None:
        result = self
    elif self.__name__ in instance.attrs:
        result = instance.attrs[self.__name__]
    else:
        result = (
            instance
            .to_padding()
            .pipe(self.__class__)
        )
        result.attrs.update(instance.attrs)
        result.instance = instance
        instance.attrs[self.__name__] = result
    return result


class Broadcast(
    SegTiles
):
    locals().update(
        __get__=__get__
    )

    @property
    def segtiles(self):
        return self.instance.segtiles

    @property
    def vectiles(self):
        return self.instance.vectiles
