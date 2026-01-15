from __future__ import annotations

import pandas as pd

from tile2net.grid.frame.namespace import namespace
from .. import frame

if False:
    from .grid import Grid


class VecTile(
    namespace,
):
    """
    Namespace for accessing vec-tile attributes aligned with in-tiles.

    See usage:
        >>> Grid.vectile
    """

    @property
    def grid(self) -> Grid:
        """Reference to parent Grid instance."""
        return self.instance

    @property
    def basegrid(self) -> Grid:
        """Reference to parent Grid instance."""
        return self.grid

    @frame.column
    def xtile(self) -> pd.Series:
        """
        X coordinate of vec-tile in vecgrid space.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.xtile
            xtile  ytile
            79320  96960    9915
        """
        grid = self.grid

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
            >>> grid: Grid
            >>> grid.seggrid.vectile.ytile
            xtile  ytile
            79320  96960    12120
        """
        grid = self.grid

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
            >>> grid: Grid
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
    def dimension(self) -> int:
        """
        Pixel dimension of each vec-tile (width and height are equal).

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.dimension
            8192
        """
        return self.grid.vecgrid.dimension

    @property
    def length(self):
        """
        Number of SegGrid tiles in one dimension of the vec-tile.

        For example, if length is 4, each vec-tile is composed of 4x4 = 16 seg-tiles.
        """
        return self.grid.vecgrid.length
