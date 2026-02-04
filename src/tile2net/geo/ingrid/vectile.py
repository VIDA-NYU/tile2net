from __future__ import annotations

import pandas as pd
from typing import TYPE_CHECKING

from tile2net.core import frame
from tile2net.core.ingrid import vectile

if TYPE_CHECKING:
    from .ingrid import InGrid


class VecTile(
    vectile.VecTile
):
    """
    Namespace for accessing vec-tile attributes aligned with in-tiles.

    See usage:
        >>> InGrid.vectile
    """

    @frame.column
    def xtile(self) -> pd.Series:
        """
        X coordinate of vec-tile in vecgrid space.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.xtile
            xtile  ytile
            79320  96960    9915
        """
        grid = self.ingrid

        vecgrid = grid.vecgrid
        length = 2 ** (grid.scale - vecgrid.scale)
        result = grid.xtile // length

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(vecgrid.xtile).all(), msg
        return result

    @frame.column
    def ytile(self) -> pd.Series:
        """
        Y coordinate of vec-tile in vecgrid space.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.ytile
            xtile  ytile
            79320  96960    12120
        """
        grid = self.ingrid

        vecgrid = grid.vecgrid
        length = 2 ** (grid.scale - vecgrid.scale)
        result = grid.ytile // length

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(vecgrid.ytile).all(), msg
        return result

    @property
    def index(self) -> pd.MultiIndex:
        """
        MultiIndex of (xtile, ytile) for vec-tiles.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.index
            MultiIndex([(9915, 12120),
                        (9915, 12120),
                        ...
                        (9915, 12120)],
                       names=['vectile.xtile', 'vectile.ytile'])
        """
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @property
    def length(self):
        """
        Number of SegGrid tiles in one dimension of the vec-tile.

        For example, if length is 4, each vec-tile is composed of 4x4 = 16 seg-tiles.
        """
        return self.ingrid.vecgrid.length
