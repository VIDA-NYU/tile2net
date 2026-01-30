from __future__ import annotations

from pathlib import Path
from typing import *

from tile2net.xyz.geocode import GeoCode
from tile2net.xyz.grid.file import File
from tile2net.xyz.grid.network import Network
from tile2net.xyz.grid.polygons import Polygons
from tile2net.xyz.grid.segtile import SegTile
from tile2net.xyz.grid.vectile import VecTile
from tile2net.xyz.util import assert_perfect_overlap

from tile2net.grid import grid
from tile2net.grid.cfg import Cfg
from tile2net.xyz import util
from tile2net.xyz.seggrid.seggrid import SegGrid
from tile2net.xyz.source.remote import Remote
from tile2net.xyz.source.source import Source
from tile2net.xyz.vecgrid.vecgrid import VecGrid

if False:
    from .filled import Filled


class Grid(
    grid.Grid
):
    """
    "Input Grid" (Grid), comprised of "input tiles" (in-tiles).
    Each tile is an image from the source.

    Example construction:
        >>> grid = Grid.from_location('Boston Common, MA')
        Grid:
                             lonmin        latmax        lonmax        latmin
        xtile  ytile
        317280 387840 -7.911538e+06  5.214840e+06 -7.911500e+06  5.214802e+06
    """

    @File
    def file(self):
        """
        Namespace container for files aligned with the tiles of a Grid.

        Example:
            >>> grid: Grid
            >>> grid.file.Static
            xtile   ytile
            317280  387840    /home/<user>/tile2net/ma/grid/static/20/31...
        """

    @VecGrid
    def vecgrid(self) -> VecGrid:
        """
        Wraps vectorization operations with a grid data structure

        After performing Grid.set_vectorization(), Grid.vecgrid is
        available for performing vectorization on the stitched tiles.

        Example:
            >>> Grid.vecgrid
            VecGrid:
                       lonmin        latmax        lonmax        latmin  \
            xtile ytile
            9915  12120 -7.911538e+06  5.214840e+06 -7.910315e+06  5.213617e+06
                                                                  geometry
            xtile ytile
            9915  12120  POLYGON ((-7910315.183 5214839.818, -7910315.1...


        Handles lazy-loading of Grid.VecGrid:
            >>> VecGrid._get

        Handles vectorization process from stitched seg-tiles:
            >>> VecGrid.vectorize
        """

    @SegGrid
    def seggrid(self) -> SegGrid:
        """
        Wraps segmentation prediction operations with a grid data structure

        After performing Grid.set_segmentation(), Grid.seggrid is
        available for performing segmentation on the stitched tiles.

        Example:
            >>> Grid.seggrid
            SegGrid:
                           lonmin        latmax        lonmax        latmin  \
            xtile ytile
            79320 96960 -7.911538e+06  5.214840e+06 -7.911385e+06  5.214687e+06
                                                                  geometry
            xtile ytile
            79320 96960  POLYGON ((-7911385.302 5214839.818, -7911385.3...
            [64 rows x 5 columns]

        Handles lazy-loading of Grid.SegGrid:
        >>> SegGrid._get

        Handles prediction of seg-tiles from stitched input tiles:
        >>> SegGrid.predict()
        """

    @classmethod
    def from_location(
            cls,
            location: Union[
                str,
                tuple[float, ...],
                list[float],
                tuple[int, ...],
                list[int]
            ],
            zoom: int = None,
            source: Union[
                str,
                Source,
                None,
                False
            ] = None,
    ) -> Self:
        """
        Instantiate a Grid from a geocoded location string or tile coordinates.

        Args:
            location: Location identifier. Can be:
                - An address string (e.g., '1600 Pennsylvania Ave, Washington DC')
                - A place name string (e.g., 'Central Park')
                - A string of 4 floats: lat,lon,lat,lon bounding box
                - A string of 4 integers: xtile,ytile,xtile,ytile bounds
                - A tuple/list of 4 floats: (lat, lon, lat, lon)
                - A tuple/list of 4 integers: (xtile, ytile, xtile, ytile)
                - Otherwise, geocoded via Nominatim
            zoom: Slippy-tile zoom level (scale) of Grid
                - Higher value -> smaller tiles
                - Typically from 14 (large area) to 20 (high detail)
                - Required when location is 4 integers (xtile, ytile bounds)
                - If not passed for lat/lon, defaults to zoom defined in cfg
            source: Tile Source instance or name/abbreviation.


        Returns:
            Grid instance covering the geocoded bounding box at the specified zoom level.

        Examples:
            Create a grid from an address. Zoom will default to that given by the Source:
            >>> grid = Grid.from_location('Times Square, New York')

            Create a grid from coordinates (lat1, lon1, lat2, lon2):
            >>> grid = Grid.from_location('42.3601,-71.0589,42.3551,-71.0539', zoom=20)

            Create a grid from tile coordinates (xtile1, ytile1, xtile2, ytile2):
            >>> grid = Grid.from_location('317280,387840,317281,387841', zoom=20)

            Create a grid from a tuple of tile coordinates:
            >>> grid = Grid.from_location((317280, 387840, 317312, 387872), zoom=20)
        """

        geocode = GeoCode.from_inferred(location, zoom)

        # resolve source and zoom
        if source is None:
            source = Source.from_inferred(geocode)
        elif source is False:
            ...
        else:
            source = Source.from_inferred(source)
        if (
                zoom is None
                and isinstance(source, Remote)
        ):
            zoom = source.zoom

        geocode = GeoCode.from_inferred(location, zoom)
        if geocode.xtile_ytile:
            out = cls.from_bounds(geocode.xtile_ytile, zoom, source)
        else:
            out = cls.from_bounds(geocode.nwse, zoom, source)

        out.location = location

        return out

    @classmethod
    def from_cfg(
            cls,
            cfg: Cfg = None
    ) -> Self:
        """
        Construct Grid from a configuration object or JSON file.

        Args:
            cfg: Configuration object, path to JSON config, or None for CLI args

        Returns:
            Fully configured Grid instance

        Example:
            >>> grid: Grid = Grid.from_cfg('config.json')
        """
        if isinstance(cfg, (str, Path)):
            cfg = Cfg.from_json(cfg)
        if cfg is None:
            cfg = Cfg.from_parser()
        with cfg:
            grid = Grid.from_location(
                location=cfg.location,
                zoom=cfg.zoom
            )
            grid.cfg = cfg

            if cfg.indir.path:
                # use input imagery
                grid = grid.set_indir()
            else:
                # set a source if specified or infer from location
                grid = grid.set_source()

            if cfg.outdir:
                grid = grid.set_outdir()

            # configure segmentation using cfg parameters
            grid = grid.set_segmentation()
            # configure vectorization using cfg parameters
            grid = grid.set_vectorization()

        return grid

    def to_cfg(
            self,
            file: Union[str, Path] = None
    ):
        raise NotImplementedError
        ...

    @classmethod
    def from_bounds(
            cls,
            bounds: Union[
                str,
                list[float],
                list[int],
                tuple[float, float, float, float],
                tuple[int, int, int, int],
                tuple[float, ...],
                tuple[int, ...],
            ],
            zoom: int = None,
            source: str | Source = None,
    ) -> Self:
        out = super().from_bounds(bounds, zoom)
        out.source = source
        if source:
            out = (
                out
                .set_segmentation()
                .set_vectorization()
            )
        return out

    @property
    def grid(self) -> Grid:
        """Quick access for the Grid of a project."""
        return self

    __name__ = 'grid'
