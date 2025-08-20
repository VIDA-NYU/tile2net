from __future__ import annotations

import numpy as np
import pandas as pd

from . import vectile
from .seggrid import SegGrid

from .. import frame, InGrid
from functools import cached_property

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
        # return self.seggrid.seggrid.vectile.length + 2


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
    instance: SegGrid

    def _get(
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
            #
            # assert instance.index.unique
        #
        # if instance is not result.instance:
        #     assert (
        #         # instance.index
        #         instance.filled.index
        #         .symmetric_difference(result.index)
        #         .empty
        #     )

        result.instance = instance
        return result

    # def _get(
    #         self: Broadcast,
    #         instance: SegGrid,
    #         owner,
    # ) -> Broadcast:
    #     if instance is None:
    #         result = self
    #     elif self.__name__ in instance.__dict__:
    #         result = instance.frame.__dict__[self.__name__]
    #     else:
    #         vecgrid = instance.vecgrid
    #         seggrid = instance.seggrid
    #         names = [
    #             seggrid.vectile.xtile.name,
    #             seggrid.vectile.ytile.name,
    #         ]
    #
    #         result = (
    #             seggrid.frame
    #             .reset_index()
    #             .set_index(names)
    #             .loc[ vecgrid.index ]
    #             .reset_index()
    #             .set_index(seggrid.frame.index.names)
    #             .pipe(Broadcast.from_frame, wrapper=instance)
    #         )
    #
    #     if instance is not result.instance:
    #         assert (
    #             # instance.index
    #             instance.filled.index
    #             .symmetric_difference(result.index)
    #             .empty
    #         )
    #
    #     result.instance = instance
    #     return result

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

    @property
    def filled(self):
        return self.instance.filled

    @property
    def ingrid(self) -> InGrid:
        return self.instance.ingrid

    # @cached_property
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
