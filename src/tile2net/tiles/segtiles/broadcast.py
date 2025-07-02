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
        tiles = self.tiles
        key = 'vectile.r'
        if key in tiles:
            return tiles[key]
        ytile = self.tiles.ytile.to_series()
        result = (
            ytile
            .groupby(self.ytile.values)
            .min()
            .loc[self.ytile]
            .rsub(ytile.values)
            .values
        )
        tiles[key] = result
        return tiles[key]

    @property
    def c(self) -> pd.Series:
        """column within the segtile of this tile"""
        tiles = self.tiles
        key = 'vectile.c'
        if key in tiles:
            return tiles[key]
        xtile = self.tiles.xtile.to_series()
        result = (
            xtile
            .groupby(self.xtile.values)
            .min()
            .loc[self.xtile]
            .rsub(xtile.values)
            .values
        )
        tiles[key] = result
        return tiles[key]



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
        vectiles = instance.vectiles
        segtiles = instance.segtiles.padded
        corners = (
            vectiles
            .to_corners(segtiles.tile.scale)
            .to_padding()
        )
        vectile = corners.index.repeat(corners.tile.area)
        kwargs = {
            'vectile.xtile': vectile.get_level_values('xtile'),
            'vectile.ytile': vectile.get_level_values('ytile'),
        }

        result = (
            corners
            .to_tiles(drop_duplicates=False)
            .assign(**kwargs)
            .pipe(Broadcast)
        )

        result.attrs.update(instance.attrs)
        instance.attrs[self.__name__] = result

        d = segtiles.tile.scale - vectiles.tile.scale
        expected = 2 ** (2 * d) + 4 * 2 ** d + 4
        assert len(result) == expected * len(vectiles)
        _ = result.vectile.r, result.vectile.c

    result.instance = instance
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

    @property
    def padded(self):
        return self.instance.padded
