from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self

import numpy as np
import pandas as pd

from . import vectile
from .seggrid import SegGrid
from ..basegrid import filled
from tile2net.grid.sampler.benchmark import Benchmark
from tile2net.grid.frame.namespace import namespace

if TYPE_CHECKING:
    pass


class VecTile(
    vectile.VecTile
):

    @property
    def row(self) -> pd.Series:
        """row within the segtile of this tile"""
        raise NotImplementedError

    @property
    def col(self) -> pd.Series:
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


class Filled(
    filled.Filled,
    SegGrid
):
    """
    SegGrid extension with additional padding tiles for VecGrid alignment.

    Automatically fills in missing seg-tiles to ensure complete coverage when
    the vectorization grid requires a different tiling scheme. Prevents gaps
    in segmentation output during vectorization.

    Handles lazy-loading of filled grid with padding:
        >>> Filled._get

    See usage:
        >>> SegGrid.filled
    """
    instance: SegGrid
    predict = False
    _predict = False

    def _get(
            self,
            instance: SegGrid,
            owner,
    ) -> Optional[Self]:
        self = namespace._get(self, instance, owner)
        cache = instance.frame.__dict__
        key = self.__name__
        if instance is None:
            return instance

        # instance = instance.seggrid
        # self.instance = instance
        if key in cache:
            result = cache[key]
        else:
            pad = instance.vectile.pad
            result: Self = (
                instance
                .to_padding(pad)
                .pipe(self.__class__)
            )

            result.__dict__.update(instance.__dict__)
            result.instance = instance

            cache[key] = result

        return result

    locals().update(
        __get__=_get
    )

    @VecTile
    def vectile(self):
        ...

    @property
    def seggrid(self):
        return self.instance.seggrid

    @property
    def sampler(self) -> Benchmark:
        return self.seggrid.sampler
