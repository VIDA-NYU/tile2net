import os
from functools import cached_property
import time
from typing import  Optional
import numpy as np
import skimage
import shapely
import geopandas as gpd
import pyproj
from shapely.geometry import MultiLineString, Polygon, shape
from pyproj import Transformer
from affine import Affine
import json
import requests
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio import features

from tile2net.raster.tile_utils.genutils import num2deg
from tile2net.raster.tile_utils.geodata_utils import _reduce_geom_precision, list_to_affine, _check_skimage_im_load

from dataclasses import dataclass, field
from typing import Union


@dataclass
class Tile:
    xtile: int
    ytile: int
    idd: int
    position: tuple
    size: int = field(default=256, repr=False)
    zoom: int = field(default=19, repr=False)
    crs: int = field(default=4326, repr=False)
    tile_step: int = field(default=1, repr=False)
    active: bool = field(default=True, repr=False)
    extension: str = 'png'


    def __post_init__(self):
        """
        Initialize the tile object
        """
        self.top = num2deg(self.xtile, self.ytile, self.zoom)[0]  # latitude of top left

        self.left = num2deg(self.xtile, self.ytile, self.zoom)[1]  # longitude of top left

        self.bottom = \
        num2deg(self.xtile + self.tile_step, self.ytile + self.tile_step, self.zoom)[
            0]  # latitude of bottom right

        self.right = \
        num2deg(self.xtile + self.tile_step, self.ytile + self.tile_step, self.zoom)[
            1]  # longitude of bottom right

        self.im_name = f'{self.xtile}_{self.ytile}.{self.extension}'
        # self.ped_poly = gpd.GeoDataFrame()
        # self.ped_poly: Optional[gpd.GeoDataFrame] = None

    @cached_property
    def ped_poly(self):
        """Return the pedestrian polygon for the tile"""
        return gpd.GeoDataFrame()

    def __hash__(self):

        return hash((self.xtile, self.ytile))

    def get_coordinates(self):
        return num2deg(self.xtile, self.ytile, self.zoom)

    def transformProject(self, src_crs, dest_crs):
        """Reproject the lat long to another crs to be used for tile acquisition.
        Args:
        ---------
            src_crs:current projection
            dest_crs:desired projection
        Returns:
        -------
            new coordinates top, left, bottom, right
        """
        transformer = Transformer.from_crs(src_crs, dest_crs)
        self.setLatlon()
        top_new, left_new = transformer.transform(self.top, self.left)
        bottom_new, right_new = transformer.transform(self.bottom, self.right)
        return top_new, left_new, bottom_new, right_new

    def setcrs(self, crs):
        pass

    def get_download_link(self, location_abr: str) -> Optional[str]:
        """
        Get the download link for the tile
        Parameters
        ----------
        location_abr: str
            Abbreviation of the location

        Returns
        -------
        Optional[str]
            The download link for the tile
        """
        # TODO: add support to check whether the data exists
        # TODO: add support to report the imagery capture date (this is easy)
        # remove any whitespaces added by mistake
        location = location_abr.strip().lower()

        # map all possible variations to standardized names
        location_map = {
            'new york city': 'nyc',
            'new york': 'ny',
            'los angles': 'la',
            'wshington dc': 'dc'
        }
        location = location_map.get(location, location)
        top, left, bottom, right = None, None, None, None
        if location == 'dc':
            assert self.base_tilesize < 512, f'Washington DC base tile size is 512, your tile size ({self.base_tilesize}) ' \
                                             f'is not compatible,  '

            top, left, bottom, right = self.transformProject(self.crs, 3857)

        dl_links = {
            'nyc': f"https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/"
                   f"NYC_Orthos_-_2020/MapServer/tile/{self.zoom}/{self.ytile}/{self.xtile}",
            'ny': f"https://orthos.its.ny.gov/arcgis/rest/services/wms/2020/MapServer/tile/"
                  f"{self.zoom}/{self.ytile}/{self.xtile}",
            'ma': f"https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/USGS_Orthos_2019/"
                  f"MapServer/tile/{self.zoom}/{self.ytile}/{self.xtile}",
            'kings': f"https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2021/"
                     f"MapServer/tile/{self.zoom}/{self.ytile}/{self.xtile}",
            'la': f'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Aerial_2014/'
                  f'MapServer/tile/{self.zoom}/{self.xtile}/{self.ytile}',
            'dc': f'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer'
                  f'/exportImage?f=image&bbox={bottom}%2C{right}%2C{top}%2C{left}'
                  f'&imageSR=102100&bboxSR=102100&size=512%2C512'

        }
        return dl_links.get(location, None)

    def dl_me(self, dest, source):
        """
        Download the tile
        Parameters
        ----------
        dest: str
        source: str

        Returns
        -------
        bool:
            True if the tile was downloaded successfully, False otherwise
        """
        output_name = f'{self.xtile}_{self.ytile}_{self.idd}.png'
        output_path = os.path.join(dest, output_name)
        time.sleep(1)  # Wait 1 second as to not overload the server
        url = self.get_download_link(location_abr=source)
        if url:
            resp = requests.get(url)
            # print("Received {} for {}".format(resp.status_code, url))
            if resp.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                return True
            else:
                print(f"{resp.content}_File could not be written")
                return False
        else:
            return False

    # @property
    def setLatlon(self):
        """Sets the lat and long of the topleft and bottomright of the tile
        Arguments
        ---------

        Returns
        ---------
        """
        self.top, self.left = num2deg(self.xtile, self.ytile, self.zoom)
        self.bottom, self.right = num2deg(self.xtile + self.tile_step,
                                          self.ytile + self.tile_step, self.zoom)

    @property
    def bbox(self):
        """
        Returns the bounding box of the tile
        Returns
        -------
        tuple
        """
        self.setLatlon()
        return self.bottom, self.top, self.left, self.right

    @property
    def tfm(self):
        """Calculate the affinity object of each tile from its bounding box

        Returns
        -------
        affinity object
        """
        self.setLatlon()
        tfm = from_bounds(self.left, self.bottom, self.right, self.top, self.size, self.size)
        return tfm

    def create_gray_image(self):
        im = np.ones((self.size, self.size)) * 50
        blck = Image.fromarray(np.uint8(im)).convert('RGB')
        return blck

    def tile2poly(self, *bounds):
        """Create a polygon geometry for the tile

        Parameters
        ----------
        bounds : tuple, optional
            A tuple containing the left, bottom, right, and top bounds of the tile.
            If not provided, the function will use the `left`, `bottom`, `right`,
            and `top` attributes of the `self` object.

        Returns
        -------
        polygon : shapely.geometry.Polygon
            A polygon geometry object representing the tile bounds.
        """
        self.setLatlon()
        if len(bounds) > 0:
            left, bottom, right, top = bounds
            poly = Polygon.from_bounds(left, bottom, right, top)
        else:
            # fix the rounding issues in plotting
            poly = Polygon.from_bounds(self.left, self.bottom - 0.00001, self.right,
                                                  self.top - 0.00001)
        # poly.set_crs(epsg=crs)
        return poly

    def tile2gdf(self, *bounds):
        """Create a tile GeoDataFrame
        Returns (GeoDataFrame): A GeoDataFrame of a single tile
        """
        if len(bounds) > 0:
            poly = self.tile2poly(*bounds)
        else:
            poly = self.tile2poly()
        tgdf = gpd.GeoDataFrame(gpd.GeoSeries(poly), columns=['geometry'], crs=self.crs)
        return tgdf


    def find_tile_neighbors_pos(self, d):
        """Returns the neighbors of a tile
        given the tile is topleft one, returns d**2-1 neighbors (d on column and d on the row)
        Parameters
        ----------
        d : int
            the size of the merge/stitch (how many tiles should be stitched on the row and columns)

        Returns
        -------
        list.
            list of the neighboring tiles (x,y)
        """
        return [[self.position[0] + r, self.position[1] + c] for r in range(0, d) for c in
                range(0, d)]

    def get_metric(self):
        """
        transform tile polygon to metric (3857) coordinate
        Args:
            tilepoly(GeoDataframe): the polygon of the tile
        Returns:
            the top,left, bottom, right coordinates of the tile
        """
        tilepoly = self.tile2gdf()
        tilepoly.to_crs(3857, inplace=True)
        # geopandas bounds method returns
        left, bottom, right, top = tilepoly.at[0, 'geometry'].bounds
        return left, top, right, bottom

    def get_individual(self, gdf):
        """Convert all multi-type geometries to single ones
        multipolygon to polygon

        Arguments
        ---------

         Returns
        -------
        """
        gdf_new = gdf.explode()
        return gdf_new

    def mask2poly(self, src_img, img_array=None):
        if img_array:
            mask_image = src_img
        else:
            mask_image = skimage.io.imread(os.path.join(src_img, f'{self.im_name}'))
        # using the masks defined here, the sidewalks are blue and hence index 2(3rd position in RGB)
        # sidewalks
        tfm_ = self.tfm
        sidewalk = mask_image[:, :, 2]
        geoms_sw = self.mask_to_poly_geojson(sidewalk)
        geoms_sw['geometry'] = geoms_sw['geometry'].apply(self.convert_poly_coords, affine_obj=tfm_)
        geoms_sw = geoms_sw.set_crs(epsg=str(self.crs))
        geoms_sw['f_type'] = 'sidewalk'

        # crosswalks
        cw = mask_image[:, :, 0]  # red channel
        geoms_cw = self.mask_to_poly_geojson(cw)
        geoms_cw = geoms_cw.set_crs(epsg=str(self.crs))
        geoms_cw['geometry'] = geoms_cw['geometry'].apply(self.convert_poly_coords, affine_obj=tfm_)
        geoms_cw['f_type'] = 'crosswalk'
        swcw = geoms_sw.append(geoms_cw)

        # Roads
        rd = mask_image[:, :, 1]  # green channel
        geoms_rd = self.mask_to_poly_geojson(rd)
        geoms_rd = geoms_rd.set_crs(epsg=str(self.crs))
        geoms_rd['geometry'] = geoms_rd['geometry'].apply(self.convert_poly_coords, affine_obj=tfm_)
        geoms_rd['f_type'] = 'road'
        rswcw = swcw.append(geoms_rd)

        rswcw.reset_index(drop=True, inplace=True)
        self.ped_poly = rswcw
        return swcw

    def get_region(self, df, spatial_index):
        """
        clips the overlapping region between the dataframe and tile extent
        Args:
            tile: the tile object to overlay
            df: the

        Returns:

        """
        tilepoly = self.tile2gdf()
        df.to_crs(tilepoly.crs, inplace=True)
        # fix rounding issues
        possible_matches_index = list(spatial_index.intersection(tilepoly.at[0, 'geometry'].bounds))

        possible_matches = df.iloc[possible_matches_index]
        region = gpd.clip(possible_matches, tilepoly)
        region.to_crs(epsg=3857, inplace=True)
        if len(region) > 0:
            return region
        else:
            return -1


    def preds_to_binary(self, pred_arr, channel_scaling=None, bg_threshold=0):
        """From Solaris
        Convert a set of predictions from a neural net to a binary mask.

        Arguments
        ---------
        pred_arr : :class:`numpy.ndarray`
            A set of predictions generated by a neural net (generally in ``float``
            dtype). This can be a 2D array or a 3D array, in which case it will
            be convered to a 2D mask output with optional channel scaling (see
            the `channel_scaling` argument). If a filename is provided instead of
            an array, the image will be loaded using scikit-image.
        channel_scaling : `list`-like of `float`s, optional
            If `pred_arr` is a 3D array, this argument defines how each channel
            will be combined to generate a binary output. channel_scaling should
            be a `list`-like of length equal to the number of channels in
            `pred_arr`. The following operation will be performed to convert the
            multi-channel prediction to a 2D output ::

                sum(pred_arr[channel]*channel_scaling[channel])

            If not provided, no scaling will be performend and channels will be
            summed.

        bg_threshold : `int` or `float`, optional
            The cutoff to set to distinguish between background and foreground
            pixels in the final binary mask. Binarization takes place *after*
            channel scaling and summation (if applicable). Defaults to 0.

        Returns
        -------
        mask_arr : :class:`numpy.ndarray`
            A 2D boolean ``numpy`` array with ``True`` for foreground pixels and
            ``False`` for background.
        """
        pred_arr = _check_skimage_im_load(pred_arr).copy()

        if len(pred_arr.shape) == 3:
            if pred_arr.shape[0] < pred_arr.shape[-1]:
                pred_arr = np.moveaxis(pred_arr, 0, -1)
            if channel_scaling is None:  # if scale values weren't provided
                channel_scaling = np.ones(shape=(pred_arr.shape[-1]),
                    dtype='float')
            pred_arr = np.sum(pred_arr * np.array(channel_scaling), axis=-1)

        mask_arr = (pred_arr > bg_threshold).astype('uint8')

        return mask_arr * 255

    def _check_crs(self, input_crs, return_rasterio=False):
        """Convert CRS to the ``pyproj.CRS`` object passed by ``solaris``."""
        if not isinstance(input_crs, pyproj.CRS) and input_crs is not None:
            out_crs = pyproj.CRS(input_crs)
        else:
            out_crs = input_crs

        if return_rasterio:
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt("WKT1_GDAL"))

        return out_crs

    def save_empty_geojson(self, path, crs):
        crs = self._check_crs(crs)
        empty_geojson_dict = {
            "type": "FeatureCollection",
            "crs":
                {
                    "type": "name",
                    "properties":
                        {
                            "name": "urn:ogc:def:crs:EPSG:{}".format(crs.to_epsg())
                        }
                },
            "features":
                []
        }

        with open(path, 'w') as f:
            json.dump(empty_geojson_dict, f)
            f.close()

    def mask_to_poly_geojson(self, pred_arr, channel_scaling=None, reference_im=None,
        min_area=40,
        bg_threshold=0, do_transform=None, simplify=True,
        tolerance=0.8, **kwargs):
        """From Solaris
        Get polygons from an image mask.

        Arguments
        ---------
        pred_arr : :class:`numpy.ndarray`
            A 2D array of integers. Multi-channel masks are not supported, and must
            be simplified before passing to this function. Can also pass an image
            file path here.
        channel_scaling : :class:`list`-like, optional
            If `pred_arr` is a 3D array, this argument defines how each channel
            will be combined to generate a binary output. channel_scaling should
            be a `list`-like of length equal to the number of channels in
            `pred_arr`. The following operation will be performed to convert the
            multi-channel prediction to a 2D output ::

                sum(pred_arr[channel]*channel_scaling[channel])

            If not provided, no scaling will be performend and channels will be
            summed.
        reference_im : str, optional
            The path to a reference geotiff to use for georeferencing the polygons
            in the mask. Required if saving to a GeoJSON (see the ``output_type``
            argument), otherwise only required if ``do_transform=True``.
        output_path : str, optional
            Path to save the output file to. If not provided, no file is saved.
        output_type : ``'csv'`` or ``'geojson'``, optional
            If ``output_path`` is provided, this argument defines what type of file
            will be generated - a CSV (``output_type='csv'``) or a geojson
            (``output_type='geojson'``).
        min_area : int, optional
            The minimum area of a polygon to retain. Filtering is done AFTER
            any coordinate transformation, and therefore will be in destination
            units.
        bg_threshold : int, optional
            The cutoff in ``mask_arr`` that denotes background (non-object).
            Defaults to ``0``.
        simplify : bool, optional
            If ``True``, will use the Douglas-Peucker algorithm to simplify edges,
            saving memory and processing time later. Defaults to ``False``.
        tolerance : float, optional
            The tolerance value to use for simplification with the Douglas-Peucker
            algorithm. Defaults to ``0.5``. Only has an effect if
            ``simplify=True``.

        Returns
        -------
        gdf : :class:`geopandas.GeoDataFrame`
            A GeoDataFrame of polygons.

        """
        from shapely.validation import make_valid
        mask_arr = self.preds_to_binary(pred_arr, channel_scaling, bg_threshold)

        if do_transform and reference_im is None:
            raise ValueError(
                'Coordinate transformation requires a reference image.')

        if do_transform:
            with rasterio.open(reference_im) as ref:
                transform = ref.transform
                crs = ref.crs
                ref.close()
        else:
            transform = Affine(1, 0, 0, 0, 1, 0)  # identity transform
            crs = rasterio.crs.CRS()

        mask = mask_arr > bg_threshold
        mask = mask.astype('uint8')

        polygon_generator = features.shapes(mask_arr,
            transform=transform,
            mask=mask)
        polygons = []
        values = []  # pixel values for the polygon in mask_arr
        for polygon, value in polygon_generator:

            p = shape(polygon).buffer(0.0)
            polygon = make_valid(p)
            if p.area >= min_area:
                polygons.append(shape(polygon).buffer(0.0))
                values.append(value)

        polygon_gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values},
            crs=crs.to_wkt())
        if simplify:
            polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(
                lambda x: x.simplify(tolerance=tolerance)
            )

        return polygon_gdf


    def get_geo_transform(self, raster_src):
        """From Solaris
        Get the geotransform for a raster image source.

        Arguments
        ---------
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
        # elif isinstance(raster_src, gdal.Dataset):
        #   affine_obj = Affine.from_gdal(*raster_src.GetGeoTransform())
        return affine_obj

    def convert_poly_coords(self, geom, raster_src=None, affine_obj=None, inverse=False,
        precision=None):
        """From Solaris
        Georegister geometry objects currently in pixel coords or vice versa.

        Arguments
        ---------
        geom : :class:`shapely.geometry.shape` or str
            A :class:`shapely.geometry.shape`, or WKT string-formatted geometry
            object currently in pixel coordinates.
        raster_src : str, optional
            Path to a raster image with georeferencing data to apply to `geom`.
            Alternatively, an opened :class:`rasterio.Band` object or
            :class:`osgeo.gdal.Dataset` object can be provided. Required if not
            using `affine_obj`.
        affine_obj: list or :class:`affine.Affine`
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


