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
        self: Broadcast,
        instance: SegTiles,
        owner,
) -> Broadcast:
    if instance is None:
        result = self
    elif self.__name__ in instance.attrs:
        result = instance.attrs[self.__name__]
    else:
        # instance.vectiles.
        vectiles = instance.vectiles
        segtiles = instance.segtiles
        corners = (
            vectiles
            .to_corners(segtiles.tile.scale)
            .to_padding()
        )
        vectile = corners.index.repeat(corners.tile.area)
        kwargs = {
            'vectile.xtile': vectile.index.get_level_values('xtile'),
            'vectile.ytile': vectile.index.get_level_values('ytile'),
        }
        result = (
            corners
            .to_tiles(drop_duplicates=False)
            .assign(**kwargs)
            .pipe(segtiles.__class__)
        )

        n = segtiles.vectile.length / segtiles.tile.length
        n += 2
        n *= n
        assert len(result) == n * len(segtiles)

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

    @VecTile
    def vectile(self):
        ...
