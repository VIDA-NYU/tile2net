from __future__ import annotations

import numpy as np
from typing import Self

from . import segtile
from .. import frame
from .grid import Grid

if False:
    from .grid import Grid
    from ..seggrid.seggrid import SegGrid
    from ..grid.grid import Grid


class SegTile(
    segtile.SegTile
):

    @frame.column
    def row(self):
        """row within the segtile of this tile"""

        ytile = self.basegrid.ytile.to_series()
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
        xtile = self.basegrid.xtile.to_series()
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
        return self.grid.seggrid.padded.length

    @frame.column
    def static(self):
        result = (
            self.basegrid.grid.seggrid.padded.static
            .loc[self.index]
            .values
        )
        return result


class Broadcast(
    Grid
):
    """
    Grid extension with broadcasting to handle overlapping seg-tiles.

    While a seg-tile consists of in-tiles, an in-tile may belong to multiple
    overlapping seg-tiles due to padding. Broadcast extends the base Grid
    to allow multiple rows per in-tile, pairing each with its corresponding
    seg-tiles.

    Handles lazy-loading of broadcast grid with padding:
        >>> Broadcast._get

    See usage:
        >>> SegGrid.broadcast
    """
    instance: Grid

    def _get(
            self,
            instance: Grid,
            owner,
    ) -> Self:
        if instance is None:
            result = self
            result.instance = instance
            return result
        instance = instance.grid
        self.instance = instance
        cache = instance.frame.__dict__
        key = self.__name__

        if key in cache:
            result = cache[key]
        else:
            # todo: why did we say we had to use seggrid broadcast?
            #   seggrid.broadcast was causing problems with `grid.groadcast.segtile.index` duplicates
            # we have to broadcast to the seggrid broacast, not just seggrid
            # seggrid = instance.seggrid.broadcast
            grid = instance.grid.filled
            pad = grid.segtile.pad
            corners = (
                instance.seggrid
                .to_corners(grid.scale)
                .to_padding(pad)
            )
            segtile = corners.index.repeat(corners.area)
            kwargs = {
                'segtile.xtile': segtile.get_level_values('xtile'),
                'segtile.ytile': segtile.get_level_values('ytile'),
            }

            result: Self = (
                corners
                .to_grid(drop_duplicates=False)
                .frame
                .assign(**kwargs)
                .pipe(self.__class__.from_frame, wrapper=instance)
            )

            instance.frame.__dict__[self.__name__] = result

            _ = result.segtile.row, result.segtile.col

        result.instance = instance

        return result

    locals().update(__get__=_get)

    @property
    def seggrid(self) -> SegGrid:
        return self.instance.seggrid

    @SegTile
    def segtile(self):
        ...

    @property
    def grid(self) -> Grid:
        return self.instance.grid

    @property
    def filled(self):
        return self

    @property
    def broadcast(self):
        return self
