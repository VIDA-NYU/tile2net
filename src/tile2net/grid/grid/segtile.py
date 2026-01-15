from __future__ import annotations

import pandas as pd

from tile2net.grid.frame.namespace import namespace
from .. import frame

if False:
    from .grid import Grid


class SegTile(
    namespace,
):
    """
    Namespace for accessing seg-tile attributes aligned with in-tiles.

    See usage:
        >>> Grid.segtile
    """
    grid: Grid

    @property
    def grid(self) -> Grid:
        """Reference to the Grid instance."""
        return self.instance

    @property
    def basegrid(self) -> Grid:
        """Reference to the parent Grid instance."""
        return self.grid

    @property
    def length(self):
        """Number of Grid tiles in one dimension of the seg-tile."""
        return self.grid.seggrid.length

    @property
    def dimension(self):
        """Pixel dimension of each seg-tile."""
        return self.grid.dimension * self.length

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of each seg-tile as (height, width)."""
        return self.dimension, self.dimension

    @property
    def index(self):
        """MultiIndex of (xtile, ytile) for seg-tiles."""
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @frame.column
    def itile(self):
        """Integer identifier for each seg-tile."""
        seggrid = self.basegrid.seggrid.broadcast
        result = (
            self.basegrid.seggrid.broadcast.itile
            .loc[~seggrid.index.duplicated()]
            .loc[self.index]
            .values
        )
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

    @frame.column
    def static(self) -> pd.Series:
        """Path to input file for this seg-tile."""
        grid = self.grid
        result = (
            grid.seggrid.filled.file.static
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def pred(self) -> pd.Series:
        """Path to grayscale segmentation file for this tile."""
        grid = self.grid
        seggrid = grid.seggrid.broadcast
        result = (
            seggrid.file.pred
            .loc[~seggrid.index.duplicated()]
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def prob(self) -> pd.Series:
        """Path to probability segmentation file for this tile."""
        grid = self.grid
        seggrid = grid.seggrid.broadcast
        result = (
            seggrid.file.prob
            .loc[~seggrid.index.duplicated()]
            .loc[self.index]
            .values
        )
        return result

    @property
    def pad(self) -> int:
        """Number of in-tiles to pad each seg-tile by."""
        return self.basegrid.cfg.segmentation.pad

    @pad.setter
    def pad(self, value: int) -> None:
        self.basegrid.cfg.segmentation.pad = value

    @pad.deleter
    def pad(self) -> None:
        del self.basegrid.cfg.segmentation.pad
