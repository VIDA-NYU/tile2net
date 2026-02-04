from __future__ import annotations

from typing import *

import pandas as pd

from tile2net.core.frame.namespace import namespace
from .. import frame

if TYPE_CHECKING:
    from .seggrid import SegGrid
    from ..vecgrid.vecgrid import VecGrid
    from ..ingrid import InGrid


class VecTile(
    namespace,
):
    """
    Namespace for accessing vec-tile attributes aligned with seg-tiles.

    See usage:
        >>> InGrid.vectile
    """

    @property
    def seggrid(self) -> SegGrid:
        """Reference to the SegGrid instance"""
        return self.instance

    @property
    def basegrid(self) -> SegGrid:
        """Reference to the parent SegGrid instance"""
        return self.seggrid

    @property
    def vecgrid(self) -> VecGrid:
        """Reference to the VecGrid instance"""
        return self.seggrid.vecgrid

    @property
    def ingrid(self) -> InGrid:
        """Reference to the Grid instance"""
        return self.seggrid.ingrid

    @property
    def length(self) -> int:
        """
        Number of SegGrid tiles in one dimension of the vec-tile.

        For example, if length is 4, each vec-tile is composed of 4x4 = 16 seg-tiles.
        """
        return self.vecgrid.length

    @property
    def dimension(self):
        """
        Pixel dimension of each vec-tile (width and height are equal).

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.dimension
            8192
        """
        return self.length * self.seggrid.dimension

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of each vec-tile as (height, width).

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.shape
            (8192, 8192)
        """
        return self.dimension, self.dimension,

    @frame.column
    def itile(self):
        """
        Integer identifier for each vec-tile.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.itile
            xtile  ytile
            79320  96960    0
        """
        result = (
            self.basegrid.vecgrid.itile
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def grayscale(self):
        """seggrid.file broadcasted to grid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.grayscale
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def static(self):
        """seggrid.file broadcasted to grid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.static
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def pred(self):
        """seggrid.file broadcasted to grid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.pred
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def prob(self):
        """seggrid.file broadcasted to grid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.prob
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def colorized(self):
        """seggrid.file broadcasted to grid"""
        vecgrid = self.vecgrid
        result = (
            vecgrid.file.colorized
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def polygon(self):
        """
        Path to polygon parquet file for this vec-tile.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.polygon
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        result = (
            self.basegrid.vecgrid.file.polygons
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def line(self):
        """
        Path to network line parquet file for this vec-tile.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.line
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        result = (
            self.basegrid.vecgrid.file.network
            .loc[self.index]
            .values
        )
        return result

    # output file paths for polygons and lines (broadcast)
    @frame.column
    def polygon_file(self):
        """
        Path to polygon parquet file for this vec-tile.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.polygon_file
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        result = (
            self.basegrid.vecgrid.file.polygons
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def network_file(self):
        """
        Path to network line parquet file for this vec-tile.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.network_file
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        result = (
            self.basegrid.vecgrid.file.network
            .loc[self.index]
            .values
        )
        return result
