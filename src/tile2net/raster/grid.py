import warnings
import os
import numpy as np
import pandas as pd
import datetime
import tempfile
from typing import Dict, Any, Union
import shapely
import rasterio
from affine import Affine
import osmnx as ox
from dataclasses import dataclass, field
from functools import cached_property
from geopy.geocoders import Nominatim

from tile2net.raster.tile_utils.topology import fill_holes, replace_convexhull
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from tile2net.raster.tile import Tile
from tile2net.raster.tile_utils.genutils import (
    deg2num, num2deg, createfolder,
)
from tile2net.raster.tile_utils.geodata_utils import (
    _reduce_geom_precision, list_to_affine, read_gdf, buff_dfs,
)
import logging
from tile2net.raster.project import Project

warnings.simplefilter(action='ignore', category=FutureWarning)
from tile2net.logger import logger


@dataclass
class GeolocateRegion:
    location: str = field(default=str)

    def mygeolocator(self) -> dict:
        app = Nominatim(user_agent="tile2net")
        return app.geocode(self.location).raw

    def get_latlon(self) -> list:
        location = self.mygeolocator()
        return list(map(float, location['boundingbox']))


@dataclass
class BaseRegion:
    name: str
    location: list[float]
    # project: Project
    crs: int = field(default=4326)

    def from_address(self):
        if isinstance(self.location, str):
            return self.round_loc(other=GeolocateRegion(self.location).get_latlon())

    def from_bounding_box(self):
        if len(self.location) == 4:
            # n, w, s, e = self.round_loc()
            # s, n, w, e = self.round_loc()
            # return s, n, w, e
            return self.round_loc()
        else:
            raise ValueError(
                'non textual address should be a list with 4 elements: '
                'topLeft_lat, topLeft_lon, bottomRight_lat, bottomRight_lon'
            )

    def round_loc(self, other: list | float = None) -> list[float]:
        if other:
            return list(np.around(np.array(other), 10))
        else:
            return list(np.around(np.array(self.location), 10))

    def __decode_address(self):
        if not isinstance(self.location, str):
            if isinstance(self.location, (list, tuple)):
                return 1
            else:
                raise TypeError(
                    f'non-textual address should be a list of coordinates, '
                    f' not {type(self.location)}'
                )
        else:
            return 2

    @cached_property
    def base_bbox(self) -> list:
        if self.__decode_address() == 1:
            return self.from_bounding_box()
        elif self.__decode_address() == 2:
            return self.from_address()
        else:
            raise ValueError('the location cannot be identified')

    @cached_property
    def base_top(self):
        return self.base_bbox[2]

    @cached_property
    def base_bottom(self):
        return self.base_bbox[0]

    @cached_property
    def base_right(self):
        return self.base_bbox[3]

    @cached_property
    def base_left(self):
        return self.base_bbox[1]

    def test_coordinates(self):
        if self.base_top < self.base_bottom:
            raise ValueError("the top and bottom is not set properly")
        if self.base_left > self.base_right:
            raise ValueError("the left and right is not set properly")
        if not (-180 < self.base_left < 180):
            raise ValueError(f"The latitude value: {self.base_left}, is not in range")
        if not (-180 < self.base_right < 180):
            raise ValueError(f"The latitude value: {self.base_right}, is not in range")
        if not (-90 < self.base_top < 90):
            raise ValueError(f"The longitude value: {self.base_top} is not in range")
        if not (-90 < self.base_bottom < 90):
            raise ValueError(f"The longitude value: {self.base_bottom} is not in range")

        logging.info('coordinates tests passed')


# perhaps freeze the base
@dataclass
class BaseGrid(BaseRegion):
    zoom: int = field(default=19)
    base_tilesize: int = field(default=256)
    padding: bool = field(default=True, repr=False)
    tile_step: int = field(default=1, repr=False)
    tiles: np.ndarray = field(default_factory=lambda: np.array(Tile), init=False, repr=False)

    def __post_init__(self):
        self.xtile = deg2num(self.base_top, self.base_left, self.zoom)[0]
        self.ytile = deg2num(self.base_top, self.base_left, self.zoom)[1]
        self.xtilem = deg2num(self.base_bottom, self.base_right, self.zoom)[0]
        self.ytilem = deg2num(self.base_bottom, self.base_right, self.zoom)[1]
        self.tiles = np.array([
            [
                Tile(
                    self.xtile + col_idx,
                    self.ytile + row_idx,
                    position=(
                        int(col_idx / self.tile_step), int(row_idx / self.stitch_step)
                    ),
                    idd=self.pos2id(
                        int(col_idx / self.tile_step), int(row_idx / self.stitch_step)
                    ),
                    zoom=self.zoom,
                    size=self.tile_size,
                    crs=self.crs
                )
                for row_idx in np.arange(0, self.base_height, self.tile_step)
            ]
            for col_idx in np.arange(0, self.base_width, self.tile_step)
        ])
        self.pose_dict = {tile.idd: tile.position for col in self.tiles for tile in col}
        # due to the rounding issues with deg2num and num2deg, we do this calculation again
        # deg2num(base_top, base_left, zoom) and then converting the nums to lat long will result in slightly different
        # lat long than original input
        self.top, self.left = num2deg(self.xtile, self.ytile, self.zoom)
        self.bottom, self.right = num2deg(self.xtilem + 1, self.ytilem + 1, self.zoom)

    @cached_property
    def base_xyrange(self) -> tuple:
        """
        calculates the min and max of xtile ytile of grid
        """
        xt, yt = deg2num(self.base_top, self.base_left, self.zoom)
        xtm, ytm = deg2num(self.base_bottom, self.base_right, self.zoom)
        return xt, yt, xtm, ytm

    @cached_property
    def base_height(self):
        return abs(self.ytilem - self.ytile) + 1  # vertical tiles (#rows)

    @cached_property
    def base_width(self):
        return abs(self.xtilem - self.xtile) + 1  # horizontal tiles (#columns)

    @cached_property
    def tile_size(self):
        return self.base_tilesize * self.stitch_step

    def pos2id(self, col_idx: int, row_idx: int):
        """
        calculate the tile id based on the position in np.array

        Parameters
        ----------
        col_idx : int
            index of the column
        row_idx : int
            index of the row

        Returns
        -------
        int
            idd of the tile
        """
        return (col_idx * self.base_height) + row_idx


@dataclass()
class Grid(BaseGrid):
    padding: bool = field(default=True, repr=False)
    tile_step: int = field(default=1, repr=False)
    stitch_step: int = field(default=1, repr=False)
    output_dir: str = field(default=os.path.join(
        tempfile.gettempdir(),
        'tile2net'
    ), repr=False)

    def __post_init__(self):
        super().__post_init__()
        # heigh/width can change if we have merging and padding
        self.height = self.base_height
        self.width = self.base_width

        self.stitched = False
        self.allow_pad = True
        self.pad = {'w': 0, 'h': 0}
        self.ntw_poly = -1
        self.sw_network = -1
        self.cw_network = -1
        self.ntw_line = -1

        self.island = -1

        self.num_inside = 0
        self.intense_buffer = False  # to be moved to class network later

        if self.padding:
            self.allow_pad = True
        # initialize the attribute values
        self.create_grid()
        self.num_active = len(self.tiles)

    def __repr__(self):
        return f"{self.name} Grid. \nCRS: {self.crs} \n" \
               f"Tile size (pixel): {self.tile_size} \nZoom level: {self.zoom} \n" \
               f"Total tiles: {self.num_tiles:,} \n" \
               f"Number of columns: {self.width:,} \n" \
               f"Number of rows: {self.height:,} \n" \
               f"xtile0: {self.xtile}, ytile0: {self.ytile} \n" \
               f"xtilem: {self.xtilem}, ytilem: {self.ytilem} \n" \
               f"stitch step: {self.tile_step}"

    def create_grid(self):
        """
        Creates the main grid based on user input

        Returns
        -------
        None.
            Nothing is returned.
        """
        self.calculate_padding()
        if self.tile_step > 1:
            self.stitched = True
            self.update_tiles()
        else:
            self.update_hw()

        # self.initialize_hw
        # handle stitching remainders
        self.num_tiles
        self.num_inside = self.num_tiles

    def calculate_padding(self):
        """
        given the tile_step or stitch_step
        calculates the width and height padding

        """
        logger.debug(f'calculate_padding()')
        if self.stitch_step == 1:
            step = self.tile_step
        else:
            step = self.stitch_step
        if step > 1:
            if not self.base_width % step == 0:
                if self.allow_pad:
                    pad = step - self.base_width % step
                    self.xtilem = self.xtilem + pad
                    self.pad['w'] = pad
                else:
                    pad = - self.base_width % step
                    self.xtilem = self.xtilem + pad
                    self.pad['w'] = pad

            if not self.base_height % step == 0:
                if self.allow_pad:
                    pad = step - self.base_height % step
                    self.ytilem = self.ytilem + pad
                    self.pad['h'] = pad
                else:
                    # if extra padding is not allowed, it means we will lose some tiles
                    # so in effect we will have negative padding
                    pad = - (self.base_height % step)
                    self.ytilem = self.ytilem + pad
                    self.pad['h'] = pad
        else:
            pass

    @cached_property
    def tile_size(self) -> int:
        """
        Returns
        -------
        int
            size of the tile in pixels
        """
        return self.base_tilesize * self.tile_step

    @property
    def bbox(self):
        """ 
        Calculates bounding box.

        Returns
        -------
        list
            minlat, maxlat, minlon, maxlon
        """
        self.top, self.left = self.round_loc(other=num2deg(self.xtile, self.ytile, self.zoom))
        self.bottom, self.right = self.round_loc(other=num2deg(self.xtilem + 1, self.ytilem + 1, self.zoom))
        return [self.bottom, self.top, self.left, self.right]

    def update_hw(self):
        """ 
        Update the height and width of the grid.
        
        Returns
        -------
        None.
            Nothing is returned.
        """

        # How many vertical tiles to cover the bbox (number of rows)
        self.height = (
                (self.base_height + self.pad['h'])
                // self.tile_step
                // self.stitch_step
                * self.stitch_step
        )
        # How many horizontal tiles to cover the bbox (number of columns)
        self.width = (
                (self.base_width + self.pad['w'])
                // self.tile_step
                // self.stitch_step
                * self.stitch_step
        )

    def update_tiles(self):
        """
        Update the tiles and their positions based on the new height/width values.
        Used when the ytile, xtile is changed in the process.

        Returns
        -------
        None.
            Nothing is returned.
        """
        logger.debug(f'update_tiles()')
        self.update_hw()
        if self.tile_step > 1:
            self.tiles = np.array([[Tile(self.xtile + col_idx,
                                         self.ytile + row_idx,
                                         position=(col_idx // self.tile_step, row_idx // self.tile_step),
                                         idd=((col_idx // self.tile_step * self.height) +
                                              row_idx // self.tile_step),
                                         zoom=self.zoom,
                                         size=self.tile_size,
                                         crs=self.crs)
                                    for row_idx in np.arange(0, self.base_height, self.tile_step)]
                                   for col_idx in np.arange(0, self.base_width, self.tile_step)])
            self.pose_dict = {tile.idd: tile.position for col in self.tiles for tile in col}

            for col_idx in range(self.width):
                for row_idx in range(self.height):
                    self.tiles[col_idx, row_idx].tile_step = self.tile_step

        else:
            self.tiles = np.array([[Tile(self.xtile + col_idx,
                                         self.ytile + row_idx,
                                         position=(col_idx, row_idx),
                                         idd=(self.pos2id(col_idx, row_idx)),
                                         zoom=self.zoom,
                                         size=self.base_tilesize * self.tile_step,
                                         crs=self.crs)
                                    for row_idx in np.arange(0, self.height)]
                                   for col_idx in np.arange(0, self.width)])
            self.pose_dict = {tile.idd: tile.position for col in self.tiles for tile in col}

    @property
    def num_tiles(self):
        return self.height * self.width

    @property
    def height_pixel(self):
        return self.height * self.tile_size

    @property
    def width_pixel(self):
        return self.width * self.tile_size

    def tilexy2pos(self, xtile, ytile):
        """
        converts xy coordinates to tile position

        Parameters
        ----------
        xtile : int
            tile.xtile
        ytile : int
            tile.ytile

        Returns
        -------
        tuple
            the position of the tile

        """
        i = xtile - self.xtile
        j = ytile - self.ytile
        return i, j

    def tilexy2id(self, xtile, ytile):
        """
        converts xy coordinates to tile idd

        Parameters
        ----------
        xtile : int
            tile.xtile
        ytile : int
            tile.ytile

        Returns
        -------
        int
            the idd of the tile

        """
        x, y = self.tilexy2pos(xtile, ytile)
        return self.pos2id(x, y)

    def _create_info_dict(self, df=False) -> dict | pd.DataFrame:
        """

        Parameters
        ----------
        df : bool
            if True, returns a DataFrame

        Returns
        -------
        dict | pd.DataFrame

        """
        tileinfo: Dict[str, Dict[Union[str, Any], Union[Union[int, str], Any]]] = {}
        for c, t in enumerate(self.tiles.flatten()):
            t.setLatlon
            if t.active:
                tileinfo.update(
                    {f'id{c}': dict(idd=c, zoom=t.zoom, xtile=t.xtile, ytile=t.ytile, topleft_x=t.left, topleft_y=t.top,
                                    bottomright_x=t.right, bottomright_y=t.bottom)})
        if df:
            tileinfo_df = pd.DataFrame.from_dict(tileinfo, orient='index')
            return tileinfo_df
        else:
            return tileinfo

    def _create_pseudo_tiles(self) -> list:
        """
        Creates the polygon representation of the grid of tiles
        ----------
        Returns
        -------
        list
            shapely polygons
        """
        poly = []
        for c, t in enumerate(self.tiles.flatten()):
            t.setLatlon
            poly.append(t.tile2poly())
        return poly

    def create_grid_gdf(self):
        """
        Creates grid's geodaframe

        Returns
        -------
        :class:`GeoDataFrame`
            Geopandas :class:`GeoDataFrame` of the grid with its tiles
        """
        tileinfo_df = self._create_info_dict(df=True)
        poly = self._create_pseudo_tiles()
        gdf_grid = gpd.GeoDataFrame(tileinfo_df, geometry=poly, crs=self.crs)
        gdf_grid = gdf_grid.reset_index(drop=True)
        return gdf_grid

    def save_shapefile(self):
        """
        Saves the grid data in shapefile format
        """
        dst_path = self.project.tiles.path
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        poly = self._create_pseudo_tiles()
        tileinfo_df = self._create_info_dict(df=True)
        gdf_grid = gpd.GeoDataFrame(tileinfo_df, geometry=poly, crs=self.crs)
        gdf_grid.to_file(
            os.path.join(dst_path, f'{self.name}_{self.tile_size}_pseudo_tiles'))

    def save_csv(self):
        """
        Saves the grid data in CSV format without geometry info
        """
        dst_path = self.project.tiles.path
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        tileinfo_df = self._create_info_dict(df=True)
        tileinfo_df.to_csv(os.path.join(dst_path, f'{self.name}_{self.tile_size}_info.csv'))

    def get_boundary(self, city=None, address=None, path=None):
        if city:
            # define the place query
            query = {'city': city}
            bound = ox.geocode_to_gdf(query)
            return bound
        elif address:
            bound = ox.geocode_to_gdf(address)
            return bound
        elif path:
            bound = read_gdf(path)
            return bound
        else:
            if isinstance(self.location, str):
                bound = ox.geocode_to_gdf(self.location)
                return bound
            else:
                raise ValueError("You must pass a textual address or name of the region")

    def get_in_boundary(self, clipped=None, city=None, address=None, path=None):
        """
        Makes the tiles outside the boundary defined by the city or address inactive. 
        This is used to speed up analysis, especially when a region has a shape that results
        in a bounding box containing many extraneous tiles.

        Only one of city, address, and path should have a value at a time. The other two should be None.

        Parameters
        ----------
        clipped : bool
            If True, returns the new pseudo tiles clipped by boundary
        city : str
            The city to create a boundary around (e.g. "Boston", "New Delhi") 
        address : str
            The address to create a boundary around (e.g. "77 Massachusetts Ave, Cambridge MA, USA") 
        path : str 
            filepath to the city/region boundary file

        Returns
        -------
        :class:`GeoDataFrame` or None
            The new pseudo tiles clipped by the boundary or None if not clipped
        """

        # create the pseudo tiles for the grid
        g = self.create_grid_gdf()
        # get the boundary shapefile
        bound = self.get_boundary(city, address, path)
        if bound.crs != self.crs:
            bound.to_crs(g.crs, inplace=True)
        # clip the pseudo tiles with boundary
        new = gpd.clip(g, bound).copy()
        self.num_inside = len(new)
        inactive = g[~(g.idd.isin(new.idd))].copy()
        self.make_inactive(list(inactive.loc[:, 'idd']))
        if clipped:
            return new

    def make_inactive(self, lst):
        """
        make the listed tiles inactive.
        Used for setting the boundaries around a region or excluding certain regions.

        Parameters
        ----------
        lst : list[int]
            list of tile ids to exclude
        """
        for pos in lst:
            self.tiles.flatten()[int(pos)].active = False
        self.num_active = self.num_tiles - len(lst)

    # noinspection PyTypeChecker
    @cached_property
    def project(self):
        return Project(
            name=self.name,
            outdir=self.output_dir,
            raster=self,
        )

    def save_ntw_polygon(self, crs_metric: int = 3857):
        """
        Collects the polygons of all tiles created in the segmentation process
        and saves them as a shapefile

        Parameters
        ----------
        crs_metric : int
            The desired coordinate reference system to save the network polygon with.
        """
        poly_fold = self.project.polygons.path
        createfolder(poly_fold)
        gdf: list[gpd.GeoDataFrame] = [
            t.ped_poly
            for t in self.tiles.flatten()
            if t.active
               and isinstance(t.ped_poly, gpd.GeoDataFrame)
               and len(t.ped_poly)
        ]
        poly_network = pd.concat(gdf)
        poly_network.reset_index(drop=True, inplace=True)
        poly_network.set_crs(self.crs, inplace=True)
        if poly_network.crs != crs_metric:
            poly_network.to_crs(crs_metric, inplace=True)
        poly_network.geometry = poly_network.simplify(0.6)
        unioned = buff_dfs(poly_network)
        unioned.geometry = unioned.geometry.simplify(0.9)
        unioned = unioned[unioned.geometry.notna()]
        unioned['geometry'] = unioned.apply(fill_holes, args=(25,), axis=1)
        simplified = replace_convexhull(unioned)
        simplified = simplified[simplified.geometry.notna()]
        simplified = simplified[['geometry', 'f_type']]
        simplified.to_crs(self.crs, inplace=True)

        self.ntw_poly = simplified
        simplified.to_file(
            os.path.join(poly_fold, f'{self.name}-Polygons-{datetime.datetime.now().strftime("%d-%m-%Y_%H")}'))
        logging.info('Polygons are generated and saved!')

    def save_ntw_polygons(
            self,
            poly_network: gpd.GeoDataFrame,
            crs_metric: int = 3857,
    ):
        """
        Collects the polygons of all tiles created in the segmentation process
        and saves them as a shapefile

        Parameters
        ----------
        poly_network : gpd.GeoDataFrame
            The concatenated GeoDataFrame formed from the polygons of each tile.
        crs_metric : int
            The desired coordinate reference system to save the network polygon with.
        """
        poly_fold = self.project.polygons.path
        createfolder(poly_fold)
        poly_network.reset_index(drop=True, inplace=True)
        poly_network.set_crs(self.crs, inplace=True)
        if poly_network.crs != crs_metric:
            poly_network.to_crs(crs_metric, inplace=True)
        poly_network.geometry = poly_network.simplify(0.6)
        unioned = buff_dfs(poly_network)
        unioned.geometry = unioned.geometry.simplify(0.9)
        unioned = unioned[unioned.geometry.notna()]
        unioned['geometry'] = unioned.apply(fill_holes, args=(25,), axis=1)
        simplified = replace_convexhull(unioned)
        simplified = simplified[simplified.geometry.notna()]
        simplified = simplified[['geometry', 'f_type']]
        simplified.to_crs(self.crs, inplace=True)

        self.ntw_poly = simplified
        path = os.path.join(poly_fold, f'{self.name}-Polygons-{datetime.datetime.now().strftime("%d-%m-%Y_%H")}')
        simplified.to_file(path)
        logging.info('Polygons are generated and saved!')

    def prepare_class_gdf(self, class_name: str, crs: int = 3857) -> object:

        """
        Parameters
        ----------
        class_name : str
            Class label, sidewalk, crosswalk, road
        
        crs : int
            The desired coordinate reference system to prepare the :class:`GeoDataFrame` with.


        Returns
        -------
        :class:`GeoDataFrame`
            class specific :class:`GeoDataFrame` in metric projection
        """
        nt = self.ntw_poly[self.ntw_poly.f_type == f'{class_name}'].copy()
        nt.geometry = nt.geometry.to_crs(crs)
        return nt

    # adopted from solaris library to overcome dependency issues
    def get_geo_transform(self, raster_src):
        """
        *Adopted from the Solaris library to overcome dependency issues*

        Get the geotransform for a raster image source.

        Parameters
        ----------
        raster_src : str, :class:`rasterio.DatasetReader`, or `osgeo.gdal.Dataset`
            Path to a raster image with georeferencing data to apply to `geom`.
            Alternatively, an opened :class:`rasterio.Band` object or
            :class:`osgeo.gdal.Dataset` object can be provided. Required if not
            using `affine_obj`.

        Returns
        -------
        transform : :class:`affine.Affine`
            An affine transformation object to the image's location in its CRS.
        """

        if isinstance(raster_src, str):
            affine_obj = rasterio.open(raster_src).transform
        elif isinstance(raster_src, rasterio.DatasetReader):
            affine_obj = raster_src.transform

        return affine_obj

    def convert_poly_coords(self, geom, raster_src=None, affine_obj=None, inverse=False,
                            precision=None):
        """
        *Adopted from the Solaris library to overcome dependency issues*

        Georegister geometry objects currently in pixel coords or vice versa.
        
        Parameters
        ----------
        raster_src : str, optional
            Path to a raster image with georeferencing data to apply to `geom`.
            Alternatively, an opened :class:`rasterio.Band` object or
            :class:`osgeo.gdal.Dataset` object can be provided. Required if not
            using `affine_obj`.
        affine_obj : list or :class:`affine.Affine`
            An affine transformation to apply to `geom` in the form of an
            ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
            Required if not using `raster_src`.
        inverse : bool, optional
            If true, will perform the inverse affine transformation, going from
            geospatial coordinates to pixel coordinates.
        precision : int, optional
            Decimal precision for the polygon output. If not provided, rounding
            is skipped.

        Returns
        -------
        out_geom
            A geometry in the same format as the input with its coordinate system
            transformed to match the destination object.
        """

        if not raster_src and not affine_obj:
            raise ValueError("Either raster_src or affine_obj must be provided.")

        if raster_src is not None:
            affine_xform = self.get_geo_transform(raster_src)
        else:
            if isinstance(affine_obj, Affine):
                affine_xform = affine_obj
            else:
                # assume it's a list in either gdal or "standard" order
                # (list_to_affine checks which it is)
                if len(affine_obj) == 9:  # if it's straight from rasterio
                    affine_obj = affine_obj[0:6]
                affine_xform = list_to_affine(affine_obj)

        if inverse:  # geo->px transform
            affine_xform = ~affine_xform

        if isinstance(geom, str):
            # get the polygon out of the wkt string
            g = shapely.wkt.loads(geom)
        elif isinstance(geom, shapely.geometry.base.BaseGeometry):
            g = geom
        else:
            raise TypeError('The provided geometry is not an accepted format. '
                            'This function can only accept WKT strings and '
                            'shapely geometries.')

        xformed_g = shapely.affinity.affine_transform(g, [affine_xform.a,
                                                          affine_xform.b,
                                                          affine_xform.d,
                                                          affine_xform.e,
                                                          affine_xform.xoff,
                                                          affine_xform.yoff])
        if isinstance(geom, str):
            # restore to wkt string format
            xformed_g = shapely.wkt.dumps(xformed_g)
        if precision is not None:
            xformed_g = _reduce_geom_precision(xformed_g, precision=precision)

        return xformed_g

    @staticmethod
    def get_exclusion_list(src_pth):
        """
        Get the list of tile ids to exclude from analysis

        Parameters
        ----------
        src_pth : str
            The file path to the csv file containing tiles to exclude

        Returns
        -------
        list[int]
            A list of integer tile ids to exclude from analysis
        """
        blacks = pd.read_csv(src_pth)
        blacks.ids = blacks.ids.astype(int)
        filtering = list(blacks['ids'])
        fint = [int(i) for i in filtering]
        return fint
