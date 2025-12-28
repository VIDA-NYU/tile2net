from __future__ import annotations
import pandas as pd

import numpy as np

from . import vectile
from .seggrid import SegGrid
from .. import frame
from ..sampler.sampler import Sampler

if False:
    from .seggrid import SegGrid
    from ..vecgrid.vecgrid import VecGrid
    from ..ingrid.ingrid import InGrid


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

    @property
    def length(self) -> int:
        return self.seggrid.vecgrid.padded.length

    @frame.column
    def grayscale(self) -> pd.Series:
        """seggrid.file broadcasted to ingrid"""
        ingrid = self.ingrid
        result = (
            ingrid.seggrid.broadcast.file.grayscale
            .loc[self.index]
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


class Broadcast(
    SegGrid
):
    """
    SegGrid extension with broadcasting to handle overlapping vec-tiles.

    While a seg-tile consists of in-tiles, an in-tile may stitch into multiple
    seg-tiles due to padding overlaps. Broadcast overcomes the base SegGrid's 
    one-row-per-tile limitation by allowing multiple entries per in-tile to
    pair them with multiple seg-tiles.

    Handles lazy-loading of broadcast grid with padding:
    >>> Broadcast._get

    See usage:
    >>> InGrid.broadcast
    """
    instance: SegGrid
    predict = False

    def _get(
            self: Broadcast,
            instance: SegGrid,
            owner,
    ) -> Broadcast:
        """
        Lazy-load factory method for accessing Broadcast from SegGrid

        Automatically generates expanded grid with padding to align segmentation
        tiles with vec-tile boundaries. Creates duplicate rows for seg-tiles that
        overlap multiple vec-tiles.

        Returns:
            Broadcast instance with expanded index including padding tiles

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.broadcast
            Broadcast with multiple rows per in-tile where overlaps occur
        """
        if instance is None:
            result = self
            result.instance = instance
            return result
        instance = instance.seggrid
        self.instance = instance
        cache = instance.frame.__dict__
        key = self.__name__

        if key in cache:
            result = cache[key]
        else:
            vecgrid = instance.vecgrid
            seggrid = instance.seggrid.filled
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

            instance.frame.__dict__[self.__name__] = result

            d = seggrid.scale - vecgrid.scale
            expected = 2 ** (2 * d) + 4 * 2 ** d + 4
            assert len(result) == expected * len(vecgrid)
            _ = result.vectile.r, result.vectile.c

        result.instance = instance
        return result

    locals().update(
        __get__=_get
    )

    @property
    def seggrid(self) -> SegGrid:
        return self.instance.seggrid

    @property
    def vecgrid(self) -> VecGrid:
        return self.instance.vecgrid

    @VecTile
    def vectile(self):
        ...

    # @property
    # def filled(self):
    #     return self.instance.filled

    @property
    def ingrid(self) -> InGrid:
        return self.instance.ingrid

    # @cached_propert
    # def dimension(self) -> int:
    #     result = self.instance.dimension
    #     result += 2 * self.ingrid.dimension
    #     return result
    #
    # @cached_property
    # def length(self) -> int:
    #     result = self.instance.length
    #     result += 2
    #     return result
    @property
    def filled(self):
        return self

    @property
    def sampler(self) -> Sampler:
        return self.seggrid.sampler

