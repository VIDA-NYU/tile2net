from __future__ import annotations
from tile2net.geo.basegrid import basegrid

from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import rasterio
import rasterio.features

from tile2net.geo.basegrid.corners import Corners
from tile2net.grid import vecgrid, frame
from tile2net.grid.cfg.cfg import Cfg

if TYPE_CHECKING:
    from ..grid import Grid


class VectorizeTask(NamedTuple):
    static: str
    affine: tuple[float, ...]
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    polygons_file: str
    network_file: str
    cfg: Cfg


class VecGrid(
    vecgrid.VecGrid,
    basegrid.BaseGrid,
):
    """
    "Vectorization Grid" (VecGrid), comprised of "vectorization tiles" (vec-tiles).
    Each vec-tile is a large tile composed of one or more SegGrid tiles, used for
    vectorizing segmentation masks into polygon and line geometries.

    VecGrid tiles are typically larger than SegGrid tiles to enable efficient
    vectorization operations. Each vec-tile covers an area equivalent to multiple
    seg-tiles, minimizing edge artifacts during polygon extraction and centerline
    network generation.

    Example:
        >>> grid: Grid
        >>> grid.vecgrid
        VecGrid:
                       lonmin        latmax        lonmax        latmin
        xtile ytile
        9915  12120 -7.911538e+06  5.214840e+06 -7.910315e+06  5.213617e+06

    VecGrid handles:
    - Grouping seg-tiles into larger vec-tiles for vectorization
    - Converting segmentation masks to vector polygons and line geometries
    - Padding tiles to reduce edge artifacts during vectorization

    See usage:
        >>> Grid.vecgrid

    Handles lazy-loading of VecGrid from Grid:
        >>> VecGrid._get
    """
    __name__ = 'vecgrid'

    @cached_property
    def padding(self) -> Corners:
        """
        Corner coordinates for padded vec-tiles.

        Computes the tile corners after applying padding, used for proper
        alignment during vectorization to avoid edge artifacts.

        Returns:
            Corners object with xmin, ymin, xmax, ymax for each padded tile

        Example:
            >>> grid: Grid
            >>> grid.vecgrid.padding
            Corners with expanded bounds for padding
        """
        result = (
            self
            .to_corners(self.seggrid.scale)
            .to_padding()
        )
        return result

    @frame.column
    def affine_params(self):
        """
        Affine transformation parameters for georeferencing each tile.

        Computes the affine transformation matrix that maps pixel coordinates
        to geographic coordinates (EPSG:3857) for each vec-tile.

        Returns:
            Series of Affine objects, one per tile

        Example:
            >>> grid: Grid
            >>> grid.vecgrid.affine_params
            xtile  ytile
            9915   12120    Affine(0.298..., 0.0, -7911538.18..., ...)
        """
        dim = self.vecgrid.padded.dimension
        padding = self.padding
        col = [
            self.lonmin.name,
            self.latmin.name,
            self.lonmax.name,
            self.latmax.name,
        ]

        it = (
            padding.frame
            [col]
            .itertuples(index=False)
        )

        result = [
            rasterio.transform
            .from_bounds(gw, gs, ge, gn, dim, dim)
            for gw, gs, ge, gn in it
        ]
        return result

    @cached_property
    def length(self) -> int:
        """
        Number of SegGrid tiles that comprise one dimension of a vec-tile.

        For example, if SegGrid uses zoom 18 and VecGrid uses zoom 16,
        each VecGrid tile is 2^2 = 4 SegGrid tiles wide.

        Example:
            >>> grid: Grid
            >>> grid.vecgrid.length
            4
        """
        result = 2 ** (self.seggrid.scale - self.scale)
        return result

    @property
    def grid(self) -> Grid:
        """
        Reference to the parent Grid instance.

        Returns:
            Grid instance that this VecGrid belongs to

        Example:
            >>> grid: Grid
            >>> vecgrid = grid.vecgrid
            >>> vecgrid.grid is grid
            True
        """
        return self.instance
