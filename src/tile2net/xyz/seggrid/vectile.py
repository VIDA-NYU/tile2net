from __future__ import annotations

import pandas as pd

from tile2net.xyz.frame.namespace import namespace
from .. import frame

if False:
    from .seggrid import SegGrid
    from ..vecgrid.vecgrid import VecGrid
    from ..grid import Grid


class VecTile(
    namespace,
):
    """
    Namespace for accessing vec-tile attributes aligned with seg-tiles.

    See usage:
        >>> Grid.vectile
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
    def grid(self) -> Grid:
        """Reference to the Grid instance"""
        return self.seggrid.grid

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
            >>> grid: Grid
            >>> grid.seggrid.vectile.dimension
            8192
        """
        return self.length * self.seggrid.dimension

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of each vec-tile as (height, width).

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.shape
            (8192, 8192)
        """
        return self.dimension, self.dimension,

    @frame.column
    def xtile(self):
        """
        X coordinate of vec-tile in vecgrid space.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.xtile
            xtile  ytile
            79320  96960    9915
        """
        seggrid = self.seggrid
        vecgrid = seggrid.vecgrid
        length = 2 ** (seggrid.scale - vecgrid.scale)
        result = seggrid.xtile // length

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(vecgrid.xtile).all(), msg
        return result

    @frame.column
    def ytile(self):
        """
        Y coordinate of vec-tile in vecgrid space.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.ytile
            xtile  ytile
            79320  96960    12120
        """
        seggrid = self.seggrid
        vecgrid = seggrid.vecgrid
        length = 2 ** (seggrid.scale - vecgrid.scale)
        result = seggrid.ytile // length

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(vecgrid.ytile).all(), msg

        return result

    @frame.column
    def itile(self):
        """
        Integer identifier for each vec-tile.

        Example:
            >>> grid: Grid
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

    @property
    def index(self):
        """
        MultiIndex of (xtile, ytile) for vec-tiles.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.index
            MultiIndex([(9915, 12120),
                        (9915, 12120)],
                       names=['vectile.xtile', 'vectile.ytile'])
        """
        # todo: we need sticky (attrs) and not-stick (__dict__)
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @frame.column
    def row(self):
        """
        Row index within the vec-tile (0 to length-1).

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.row
            xtile  ytile
            79320  96960    0
                   96961    1
                   96962    2
                   96963    3
                   96964    4
                   96965    5
                   96966    6
        """

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
        """
        Column index within the vec-tile (0 to length-1).

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.col
            xtile  ytile
            79320  96960    0
                   96961    0
            79327  96967    7
        """
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

    @frame.column
    def polygon(self):
        """
        Path to polygon parquet file for this vec-tile.

        Example:
            >>> grid: Grid
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
            >>> grid: Grid
            >>> grid.seggrid.vectile.line
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        result = (
            self.basegrid.vecgrid.file.lines
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def affine(self):
        """
        Affine transformation parameters for georeferencing each vec-tile.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.affine
            xtile  ytile
            79320  96960    Affine(1.341104507446289e-06, 0.0, -71.0719299316406...
        """
        vecgrid = self.vecgrid
        result = (
            vecgrid.affine_params
            .loc[self.index]
            .values
        )
        return result

    # west/east/south/north bounds in projected coords (broadcast)
    @frame.column
    def lonmin(self):
        """
        Western boundary of vec-tile in projected coordinates (EPSG:3857).
        
        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.lonmin
            xtile  ytile
            79320  96960    -71.0719299316406
        """
        result = (
            self.vecgrid.lonmin
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def latmin(self):
        """
        Southern boundary of vec-tile in projected coordinates (EPSG:3857).

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.latmin
            xtile  ytile
            79320  96960    5.213617e+06
        """
        result = (
            self.vecgrid.latmin
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def lonmax(self):
        """
        Eastern boundary of vec-tile in projected coordinates (EPSG:3857).

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.lonmax
            xtile  ytile
            79320  96960   -7.910315e+06
        """
        result = (
            self.vecgrid.lonmax
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def latmax(self):
        """
        Northern boundary of vec-tile in projected coordinates (EPSG:3857).

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.latmax
            xtile  ytile
            79320  96960    5.214840e+06
        """
        result = (
            self.vecgrid.latmax
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
            >>> grid: Grid
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
            >>> grid: Grid
            >>> grid.seggrid.vectile.network_file
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        result = (
            self.basegrid.vecgrid.file.lines
            .loc[self.index]
            .values
        )
        return result

    @property
    def pad(self) -> int:
        """
        Padding pixels applied to vec-tiles during vectorization.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.pad
            1
        """
        return self.basegrid.cfg.vectorization.pad

    @pad.setter
    def pad(self, value: int) -> None:
        self.basegrid.cfg.vectorization.pad = value

    @pad.deleter
    def pad(self) -> None:
        del self.basegrid.cfg.vectorization.pad
