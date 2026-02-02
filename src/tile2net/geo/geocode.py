from __future__ import annotations

import functools
import os.path
import re
from functools import cached_property
from pathlib import Path
from typing import *

import geopandas as gpd
import geopy
import osmnx
import shapely
from geopandas import GeoDataFrame, GeoSeries
from geopy.geocoders import Nominatim, Photon

from tile2net.geo import util
from tile2net.logger import logger


class cached:
    def __set__(self, instance, value):
        instance.__dict__[self.__name__] = value

    def __init__(self, func):
        self.__func__ = func

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __delete__(self, instance):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            pass

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return instance.__dict__[self.__name__]
        except KeyError:
            result = instance.__dict__[self.__name__] = (
                self.__func__
                .__get__(instance, owner)
                .__call__()
            )
            return result


class GeoCode:
    cache: dict[object, GeoCode] = {}
    _round = staticmethod(functools.partial(round, ndigits=4))
    passed: object
    """Originally passed object to construct the class."""

    @staticmethod
    def _match(obj: str) -> tuple[str, str, str, str]:
        """
        Parses a string for exactly 4 numeric coordinates (integers or floats).
        Handles formats like "1, 2, 3, 4", "(1.1, 2.2, 3.3, 4.4)", or "minx=1..."
        """
        pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
        matches = re.findall(pattern, obj)

        if len(matches) != 4:
            raise ValueError(f"Expected 4 coordinates, found {len(matches)} in '{obj}'")

        return matches[0], matches[1], matches[2], matches[3]

    @classmethod
    def from_inferred(
            cls,
            obj: str | Any,
            zoom: Optional[int] = None
    ) -> Self:
        original_obj = obj

        if isinstance(obj, str):
            try:
                obj = cls._match(obj)
            except ValueError:
                out = cls.from_address(obj)
                out.passed = original_obj
                return out
            else:
                out = cls.from_inferred(obj)
                out.passed = original_obj
                return out

        if (
                isinstance(obj, (list, tuple))
                and len(obj) == 2
        ):
            out = cls.from_centroid(obj)
            out.passed = original_obj
            return out
        if (
                isinstance(obj, (list, tuple))
        ):
            if any(
                    isinstance(v, float)
                    for v in obj
            ):
                obj = list(map(float, obj))
            elif any(
                    isinstance(v, int)
                    for v in obj
            ):
                try:
                    obj = list(map(int, obj))
                except ValueError:
                    obj = list(map(float, obj))
            else:
                obj = list(map(float, obj))

            out = cls.from_latlon(obj)
            out.passed = original_obj
            return out
        if isinstance(obj, shapely.Polygon):
            out = cls.from_polygon(obj)
            out.passed = original_obj
            return out
        if isinstance(obj, shapely.geometry.Point):
            out = cls.from_point(obj)
            out.passed = original_obj
            return out
        raise ValueError(f'Could not infer geocode from {obj}')

    @classmethod
    def from_polygon(cls, polygon: shapely.Polygon) -> Self:
        if polygon.bounds in cls.cache:
            return cls.cache[polygon.bounds]
        result = cls()
        result.passed = polygon
        result.polygon = polygon
        w, s, e, n = polygon.bounds
        result.nwse = n, w, s, e
        cls.cache[polygon.bounds] = result
        return result

    @classmethod
    def from_address(cls, address: str) -> Self:
        if address in cls.cache:
            return cls.cache[address]
        result = cls()
        result.passed = address
        result.address = address
        cls.cache[address] = result
        return result

    @classmethod
    def from_point(cls, point: shapely.geometry.Point) -> Self:
        centroid = (cls._round(point.y), cls._round(point.x))
        if centroid in cls.cache:
            return cls.cache[centroid]
        result = cls()
        result.passed = point
        result.centroid = centroid
        cls.cache[centroid] = result
        return result

    @classmethod
    def from_lonlat(
            cls,
            lonlat: Union[
                list[float],
                tuple[int, ...]
            ]
    ) -> Self:
        passed = lonlat
        lonlat = tuple(map(cls._round, lonlat))
        if lonlat in cls.cache:
            return cls.cache[lonlat]
        result = cls()
        result.passed = passed
        w, e = min(lonlat[0], lonlat[2]), max(lonlat[0], lonlat[2])
        s, n = min(lonlat[1], lonlat[3]), max(lonlat[1], lonlat[3])
        result.nwse = n, w, s, e
        cls.cache[lonlat] = result
        return result

    @classmethod
    def from_latlon(
            cls,
            latlon: Union[
                list[float],
                tuple[int, ...]
            ]
    ) -> Self:
        lonlat = [latlon[1], latlon[0], latlon[3], latlon[2]]
        return cls.from_lonlat(lonlat)

    @classmethod
    def from_centroid(cls, centroid: tuple[float, float] | list[float]) -> Self:
        original_centroid = centroid
        centroid = tuple(centroid)
        if centroid in cls.cache:
            return cls.cache[centroid]
        result = cls()
        result.passed = original_centroid
        result.centroid = centroid
        cls.cache[centroid] = result
        return result

    @classmethod
    def from_geometry(
            cls,
            geometry: str
    ) -> Self:
        original_geometry = geometry
        result = cls()
        result.passed = original_geometry
        if (
                isinstance(geometry, (Path, str))
                and '.' in geometry
        ):
            match str(geometry).split('.')[-1]:
                case 'feather':
                    geometry = gpd.read_feather(geometry)
                case 'parquet':
                    geometry = gpd.read_parquet(geometry)
                case _:
                    geometry = gpd.read_file(geometry)
        elif isinstance(geometry, str):
            geometry = osmnx.geocode(geometry)
        elif isinstance(geometry, (GeoSeries, GeoDataFrame)):
            result.geometry = GeoDataFrame(geometry)
        else:
            raise ValueError(
                f"Could not infer geometry from '{geometry}'"
            )
        result.geometry = geometry
        return result

    @classmethod
    def from_osm(cls, query: str):
        geometry = osmnx.geocode_to_gdf(query)
        result = cls.from_frame(geometry)
        result.passed = query
        return result

    @classmethod
    def from_frame(cls, frame: GeoDataFrame | str | Path) -> Self:
        original_frame = frame
        if isinstance(frame, Path):
            frame = str(frame)
        if isinstance(frame, str):
            match frame.split('.')[-1]:
                case 'feather':
                    frame = gpd.read_feather(frame)
                case 'parquet':
                    frame = gpd.read_parquet(frame)
                case _:
                    frame = gpd.read_file(frame)
        elif isinstance(frame, (GeoDataFrame, GeoSeries)):
            frame = GeoDataFrame(frame)
        else:
            raise ValueError(f"Could not infer frame from '{frame}'")
        result = cls()
        result.passed = original_frame
        result.geometry = frame
        return result

    @classmethod
    def from_xtile_ytile(
            cls,
            xtile_ytile: tuple[int, ...],
            zoom: int
    ) -> Self:
        lonlat = (
            *util.xy2lonlat(*xtile_ytile[:2], zoom),
            *util.xy2lonlat(*xtile_ytile[2:], zoom)
        )
        result = cls.from_lonlat(lonlat)
        result.passed = xtile_ytile
        result.xtile_ytile = tuple(xtile_ytile)
        result.zoom = zoom
        return result

    @cached_property
    def nwse(self) -> tuple[float, ...]:
        msg = f'Geocoding the following, please wait:\n \t{self.address}'
        logger.info(msg)

        result: geopy.Location = None
        geocoder = None
        try:
            result = (
                Nominatim(user_agent='tile2net')
                .geocode(self.address, timeout=10, addressdetails=True)
            )
        except Exception as e:
            logger.debug(f"Nominatim geocoding failed: {e}, trying Photon")
        else:
            geocoder = 'nominatim'
            tags = result.raw['address']
            self.tags = tags

        if result is None:
            try:
                result = (
                    Photon(user_agent='tile2net')
                    .geocode(self.address, timeout=10)
                )
            except Exception as e:
                logger.error(f"Photon geocoding also failed: {e}")
                raise ValueError(f"Could not geocode '{self.address}'") from e
            else:
                geocoder = 'photon'
                tags = result.raw['properties']
                self.tags = tags

        if result is None:
            raise ValueError(f"Could not geocode '{self.address}'")

        if geocoder == 'nominatim':
            bbox = result.raw['boundingbox']
            it = (float(x) for x in bbox)
            s, n, w, e = (self._round(x) for x in it)
        else:
            extent = result.raw['properties']['extent']
            w, s, e, n = map(self._round, extent)

        return n, w, s, e

    @cached_property
    def tags(self):
        """
        keywords
            {'osm_type': 'W',
             'osm_id': 427818536,
             'osm_key': 'leisure',
             'osm_value': 'park',
             'type': 'other',
             'countrycode': 'US',
             'name': 'Central Park',
             'country': 'United States',
             'city': 'New York',
             'district': 'New York County',
             'state': 'NY',
             'extent': [-73.9814075, 40.8003135, -73.9496061, 40.7647275]}
        """
        # todo: make this a proper cached_property instead of having nwse and address set it
        _ = self.nwse
        _ = self.address
        return self.tags

    @cached_property
    def wsen(self):
        n, w, s, e = self.nwse
        return w, s, e, n

    @cached_property
    def address(self) -> str:
        result: geopy.Location = None
        try:
            result = (
                Nominatim(user_agent='tile2net')
                .reverse(self.centroid, timeout=10)
            )
            keywords = result.raw['address']
        except Exception as e:
            logger.debug(f"Nominatim reverse geocoding failed: {e}, trying Photon")

        if result is None:
            try:
                result = (
                    Photon(user_agent='tile2net')
                    .reverse(self.centroid, timeout=10)
                )
                keywords = result.raw['properties']
            except Exception as e:
                logger.error(f"Photon reverse geocoding also failed: {e}")
                raise ValueError(f"Could not reverse geocode '{self.centroid=}'") from e

        if result is None:
            raise ValueError(f"Could not reverse geocode '{self.centroid=}'")

        self.tags = keywords
        out = result.address
        return out

    @cached_property
    def centroid(self) -> tuple[float, float]:
        """Centroid as (lat, lon)"""
        bounds = self.nwse
        y = (bounds[0] + bounds[2]) / 2
        x = (bounds[1] + bounds[3]) / 2
        return y, x

    @cached_property
    def name(self) -> str:
        result = os.path.normcase(
            self.address
            .split(',')
            [0]
            .casefold()
        )
        return result

    @cached_property
    def polygon(self) -> shapely.Polygon:
        result = shapely.geometry.box(*self.wsen)
        return result

    @cached_property
    def geometry(self) -> GeoSeries:
        return osmnx.geocode_to_gdf(self.address)

    @cached_property
    def xtile_ytile(self) -> Optional[tuple[int, int, int, int]]:
        """
        Original xtile_ytile bounds if geocode was created from tile coordinates.
        Format: (xmin, ymin, xmax, ymax)

        See:
            >>> GeoCode.from_xtile_ytile()
        """
        return

    @cached_property
    def zoom(self) -> Optional[int]:
        """
        Zoom level if geocode was created from tile coordinates.
        See:
            >>> GeoCode.from_xtile_ytile()
        """
        return

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        items = [
            f"    {k}={v!r},"
            for k, v in self.__dict__.items()
        ]
        params = "\n".join(items)
        out = f"{cls_name}(\n{params}\n)"
        return out


if __name__ == '__main__':
    # GeoCode.from_geometry('New York City')
    # GeoCode.from_geometry('Washington Square Park, New York, NY, USA')
    # GeoCode.from_geometry('40.70661915280362, -74.01066228152449')
    # GeoCode.from_geometry('55 Exchange Pl #5, New York, NY 10005')
    GeoCode.from_osm('New York City')
    GeoCode.from_osm('Washington Square Park, New York, NY, USA')
    GeoCode.from_osm('40.70661915280362, -74.01066228152449')
    GeoCode.from_osm('55 Exchange Pl #5, New York, NY 10005')
