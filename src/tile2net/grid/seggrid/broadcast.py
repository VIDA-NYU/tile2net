from __future__ import annotations

import numpy as np
import pandas as pd

from . import vectile
from .seggrid import SegGrid

from .. import frame


class VecTile(
    vectile.VecTile
):

    @frame.column
    def r(self):
        """row within the segtile of this tile"""

        ytile = self.grid.ytile.to_series()
        result = (
            ytile
            .groupby(self.ytile.values)
            .min()
            .loc[self.ytile]
            .rsub(ytile.values)
            .values
        )
        return result

    @frame.column
    def c(self):
        """column within the segtile of this tile"""
        xtile = self.grid.xtile.to_series()
        result = (
            xtile
            .groupby(self.xtile.values)
            .min()
            .loc[self.xtile]
            .rsub(xtile.values)
            .values
        )
        return result



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
        self: Broadcast,
        instance: SegGrid,
        owner,
) -> Broadcast:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.frame.__dict__[self.__name__]
    else:
        vecgrid = instance.vecgrid
        seggrid = instance.seggrid.padded
        corners = (
            vecgrid
            .to_corners(seggrid.scale)
            .to_padding()
        )
        vectile = corners.index.repeat(corners.area)
        kwargs = {
            'vectile.xtile': vectile.get_level_values('xtile'),
            'vectile.ytile': vectile.get_level_values('ytile'),
        }

        result = (
            corners
            .to_grid(drop_duplicates=False)
            .frame
            .assign(**kwargs)
            .pipe(Broadcast.from_frame, wrapper=instance)
        )

        # instance.__dict__[self.__name__] = result
        instance.frame.__dict__[self.__name__] = result

        d = seggrid.scale - vecgrid.scale
        expected = 2 ** (2 * d) + 4 * 2 ** d + 4
        assert len(result) == expected * len(vecgrid)
        _ = result.vectile.r, result.vectile.c

    result.instance = instance
    return result

class Broadcast(
    SegGrid
):
    locals().update(
        __get__=__get__
    )

    @property
    def seggrid(self):
        return self.instance.seggrid

    @property
    def vecgrid(self):
        return self.instance.vecgrid

    @VecTile
    def vectile(self):
        ...

    @property
    def padded(self):
        return self.instance.padded
