from __future__ import annotations

import numpy as np
import pandas as pd

from . import vectile
from .seggrid import SegGrid
from ..grid import padded


class VecTile(
    vectile.VecTile
):

    @property
    def r(self) -> pd.Series:
        """row within the segtile of this tile"""
        raise NotImplementedError

    @property
    def c(self) -> pd.Series:
        """column within the segtile of this tile"""
        raise NotImplementedError


def boundary_grid(
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
        self: Padded,
        instance: SegGrid,
        owner,
) -> Padded:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        result = (
            instance
            .to_padding()
            .pipe(self.__class__)
        )
        assert instance.index.isin(result.index).all()
        result.__dict__.update(instance.__dict__)
        instance.__dict__[self.__name__] = result

    result.instance = instance
    return result


class Padded(
    padded.Padded,
    SegGrid
):
    locals().update(
        __get__=__get__
    )

    @VecTile
    def vectile(self):
        ...
