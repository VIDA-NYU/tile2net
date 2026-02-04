from __future__ import annotations

from typing import *

import pandas as pd

from tile2net.core.frame.namespace import namespace
from .. import frame

if TYPE_CHECKING:
    from .ingrid import InGrid


class SegTile(
    namespace,
):
    """
    Namespace for accessing seg-tile attributes aligned with in-tiles.

    See usage:
        >>> InGrid.segtile
    """

    @property
    def ingrid(self) -> InGrid:
        """Reference to the Grid instance."""
        return self.instance

    @property
    def grid(self) -> InGrid:
        """Reference to the parent Grid instance."""
        return self.ingrid

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
        result = pd.Index(self.itile, name='itile')
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
    def static(self) -> pd.Series:
        """Path to input file for this seg-tile."""
        grid = self.ingrid
        result = (
            grid.seggrid.filled.file.static
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def pred(self) -> pd.Series:
        """Path to grayscale segmentation file for this tile."""
        grid = self.ingrid
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
        grid = self.ingrid
        seggrid = grid.seggrid.broadcast
        result = (
            seggrid.file.prob
            .loc[~seggrid.index.duplicated()]
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def unclipped_prob(self) -> pd.Series:
        """Path to unclipped probability segmentation file for this tile."""
        grid = self.ingrid
        seggrid = grid.seggrid.broadcast
        result = (
            seggrid.file.unclipped_prob
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
