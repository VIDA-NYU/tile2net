from __future__ import annotations

import numpy as np

from . import segtile
from .. import frame
from .ingrid import InGrid

if False:
    from .ingrid import InGrid
    from ..seggrid.seggrid import SegGrid
    from ..ingrid.ingrid import InGrid


class SegTile(
    segtile.SegTile
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
        return self.ingrid.seggrid.padded.length

    @frame.column
    def infile(self):
        result = (
            self.grid.ingrid.seggrid.padded.infile
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
    InGrid
):
    instance: InGrid

    def _get(
            self: Broadcast,
            instance: InGrid,
            owner,
    ) -> Broadcast:
        if instance is None:
            result = self
        elif self.__name__ in instance.__dict__:
            result = instance.frame.__dict__[self.__name__]
        else:
            seggrid = instance.seggrid
            ingrid = instance.ingrid.filled
            corners = (
                seggrid
                .to_corners(ingrid.scale)
                .to_padding()
            )
            segtile = corners.index.repeat(corners.area)
            kwargs = {
                'segtile.xtile': segtile.get_level_values('xtile'),
                'segtile.ytile': segtile.get_level_values('ytile'),
            }

            result = (
                corners
                .to_grid(drop_duplicates=False)
                .frame
                .assign(**kwargs)
                .pipe(Broadcast.from_frame, wrapper=instance)
            )

            instance.frame.__dict__[self.__name__] = result

            d = ingrid.scale - seggrid.scale
            expected = 2 ** (2 * d) + 4 * 2 ** d + 4
            assert len(result) == expected * len(seggrid)
            _ = result.segtile.r, result.segtile.c

        result.instance = instance
        return result

    locals().update(
        __get__=_get
    )

    @property
    def ingrid(self) -> InGrid:
        return self.instance.ingrid

    @property
    def seggrid(self) -> SegGrid:
        return self.instance.seggrid

    @SegTile
    def segtile(self):
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
