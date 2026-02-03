from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tile2net.core.frame import frame
from tile2net.core.grid import segtile

if TYPE_CHECKING:
    from .grid import Grid


class SegTile(
    segtile.SegTile
):
    """
    Namespace for accessing seg-tile attributes aligned with in-tiles.

    See usage:
        >>> Grid.segtile
    """
    grid: Grid

    @property
    def index(self):
        """MultiIndex of (xtile, ytile) for seg-tiles."""
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @frame.column
    def xtile(self):
        """X tile coordinate of the SegGrid tile associated with the Grid tile."""
        grid = self.grid

        seggrid = grid.seggrid.filled
        result: pd.Index = grid.xtile.__floordiv__(grid.segtile.length)

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(seggrid.xtile).all(), msg
        return result

    @frame.column
    def ytile(self):
        """Y tile coordinate of the SegGrid tile associated with the Grid tile."""
        grid = self.grid

        seggrid = grid.seggrid.filled
        result: pd.Index = grid.ytile.__floordiv__(grid.segtile.length)

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(seggrid.ytile).all(), msg
        return result

    @frame.column
    def row(self) -> pd.Series:
        """Row index within the seg-tile (0 to length-1)."""
        grid = self.grid
        result = (
            grid.ytile
            .to_series(index=grid.index)
            .floordiv(grid.segtile.length)
            .mul(grid.segtile.length)
            .rsub(grid.ytile.values)
        )
        return result

    @frame.column
    def col(self) -> pd.Series:
        """Column index within the seg-tile (0 to length-1)."""
        grid = self.grid
        result = (
            grid.xtile
            .to_series(index=grid.index)
            .floordiv(grid.segtile.length)
            .mul(grid.segtile.length)
            .rsub(grid.xtile.values)
        )
        return result

    @property
    def length(self):
        """
        Number of SegGrid tiles in one dimension of the vec-tile.

        For example, if length is 4, each vec-tile is composed of 4x4 = 16 seg-tiles.
        """
        return self.grid.seggrid.length
