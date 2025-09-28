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
            pad = seggrid.pad
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
            _ = result.segtile.r, result.segtile.c

        result.instance = instance
        return result

    locals().update(
        __get__=_get
    )

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
