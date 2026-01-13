from __future__ import annotations

import numpy as np
from typing import Self

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
    def row(self):
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
    def col(self):
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
    def static(self):
        result = (
            self.grid.ingrid.seggrid.padded.static
            .loc[self.index]
            .values
        )
        return result


class Broadcast(
    InGrid
):
    """
    InGrid extension with broadcasting to handle overlapping seg-tiles.

    While a seg-tile consists of in-tiles, an in-tile may belong to multiple
    overlapping seg-tiles due to padding. Broadcast extends the base InGrid
    to allow multiple rows per in-tile, pairing each with its corresponding
    seg-tiles.

    Handles lazy-loading of broadcast grid with padding:
        >>> Broadcast._get

    See usage:
        >>> SegGrid.broadcast
    """
    instance: InGrid

    def _get(
            self,
            instance: InGrid,
            owner,
    ) -> Self:
        if instance is None:
            result = self
            result.instance = instance
            return result
        instance = instance.ingrid
        self.instance = instance
        cache = instance.frame.__dict__
        key = self.__name__

        if key in cache:
            result = cache[key]
        else:
            # we have to broadcast to the seggrid broacast, not just seggrid
            seggrid = instance.seggrid.broadcast
            ingrid = instance.ingrid.filled
            pad = ingrid.segtile.pad
            corners = (
                seggrid
                .to_corners(ingrid.scale)
                .to_padding(pad)
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
            # expected = 2 ** (2 * d) + 4 * 2 ** d + 4
            # expected = 2 ** (2 * d * pad) + 4 * 2 ** (d * pad) + 4
            # assert len(result) == expected * len(seggrid)
            _ = result.segtile.row, result.segtile.col

        result.instance = instance
        return result

    locals().update( __get__=_get )

    @property
    def seggrid(self) -> SegGrid:
        return self.instance.seggrid

    @SegTile
    def segtile(self):
        ...

    @property
    def ingrid(self) -> InGrid:
        return self.instance.ingrid

    @property
    def filled(self):
        return self

    @property
    def broadcast(self):
        return self
