from __future__ import annotations

import pandas as pd

from tile2net.grid.frame.namespace import namespace
from .. import frame

if False:
    from .ingrid import InGrid


class SegTile(
    namespace,
):
    """
    Namespace for accessing seg-tile attributes aligned with in-tiles.

    See usage:
        >>> InGrid.segtile
    """
    ingrid: InGrid

    @property
    def ingrid(self) -> InGrid:
        """Reference to the InGrid instance."""
        return self.instance

    @property
    def grid(self) -> InGrid:
        """Reference to the parent InGrid instance."""
        return self.ingrid

    @property
    def length(self):
        """Number of InGrid tiles in one dimension of the seg-tile."""
        return self.ingrid.seggrid.length

    @property
    def dimension(self):
        """Pixel dimension of each seg-tile."""
        return self.ingrid.dimension * self.length

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
        seggrid = self.grid.seggrid.broadcast
        result = (
            self.grid.seggrid.broadcast.itile
            .loc[~seggrid.index.duplicated()]
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def xtile(self):
        """X tile coordinate of the SegGrid tile associated with the InGrid tile."""
        ingrid = self.ingrid

        seggrid = ingrid.seggrid.filled
        result: pd.Index = ingrid.xtile.__floordiv__(ingrid.segtile.length)

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(seggrid.xtile).all(), msg
        return result

    @frame.column
    def ytile(self):
        """Y tile coordinate of the SegGrid tile associated with the InGrid tile."""
        ingrid = self.ingrid

        seggrid = ingrid.seggrid.filled
        result: pd.Index = ingrid.ytile.__floordiv__(ingrid.segtile.length)

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(seggrid.ytile).all(), msg
        return result

    @frame.column
    def row(self) -> pd.Series:
        """Row index within the seg-tile (0 to length-1)."""
        ingrid = self.ingrid
        result = (
            ingrid.ytile
            .to_series(index=ingrid.index)
            .floordiv(ingrid.segtile.length)
            .mul(ingrid.segtile.length)
            .rsub(ingrid.ytile.values)
        )
        return result

    @frame.column
    def col(self) -> pd.Series:
        """Column index within the seg-tile (0 to length-1)."""
        ingrid = self.ingrid
        result = (
            ingrid.xtile
            .to_series(index=ingrid.index)
            .floordiv(ingrid.segtile.length)
            .mul(ingrid.segtile.length)
            .rsub(ingrid.xtile.values)
        )
        return result

    @frame.column
    def infile(self) -> pd.Series:
        """Path to input file for this seg-tile."""
        ingrid = self.ingrid
        result = (
            ingrid.seggrid.filled.file.infile
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def pred(self) -> pd.Series:
        """Path to grayscale segmentation file for this tile."""
        ingrid = self.ingrid
        seggrid = ingrid.seggrid.broadcast
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
        ingrid = self.ingrid
        seggrid = ingrid.seggrid.broadcast
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
        return self.grid.cfg.segmentation.pad

    @pad.setter
    def pad(self, value: int) -> None:
        self.grid.cfg.segmentation.pad = value

    @pad.deleter
    def pad(self) -> None:
        del self.grid.cfg.segmentation.pad
