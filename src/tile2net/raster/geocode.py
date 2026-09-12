from __future__ import annotations
from functools import cached_property

import functools
import os.path

from pathlib import Path
from typing import *

import geopandas as gpd
import osmnx
import shapely
from geopandas import GeoDataFrame, GeoSeries
from geopy.geocoders import Photon

from tile2net.logger import logger
from tile2net.raster.nominatim import Nominatim


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
    _round = functools.partial(round, ndigits=4)

    def __repr__(self) -> str:
        fields = ',\n    '.join(
            f'{name}={self.__dict__[name]!r}'
            for name in ('address', 'centroid', 'nwse')
            if name in self.__dict__
        )
        if not fields:
            return f'{type(self).__name__}()'
        return f'{type(self).__name__}(\n    {fields}\n)'

    @classmethod
    def from_inferred(cls, obj: str | Any) -> Self:
        if isinstance(obj, shapely.Polygon):
            return cls.from_polygon(obj)
        if isinstance(obj, str):
            try:
                bounds = [
                    float(num)
                    for num in obj.split(',')
                ]
                match len(bounds):
                    case 2:
                        return cls.from_centroid(bounds)
                    case 4:
                        return cls.from_nwse(bounds)
                    case _:
                        raise ValueError(
                            f"Could not infer geocode from '{obj}'"
                        )
            except (ValueError, AttributeError):
                return cls.from_address(obj)
        if isinstance(obj, (list, tuple)):
            if len(obj) == 2:
                return cls.from_centroid(obj)
            elif len(obj) == 4:
                return cls.from_nwse(obj)
        raise ValueError(
            f"Could not infer geocode from '{obj}'"
        )

        return result

    @classmethod
    def from_polygon(cls, polygon: shapely.Polygon) -> Self:
        if polygon.bounds in cls.cache:
            return cls.cache[polygon.bounds]
        result = cls()
        result.polygon = polygon
        w, s, e, n = polygon.bounds
        result.nwse = n, w, s, e
        return result

    @classmethod
    def from_address(cls, address: str) -> Self:
        if address in cls.cache:
            return cls.cache[address]
        result = cls()
        result.address = address
        cls.cache[address] = result
        return result

    @classmethod
    def from_nwse(cls, bounds: list[float]) -> Self:
        bounds = tuple(map(cls._round, bounds))
        if bounds in cls.cache:
            return cls.cache[bounds]
        result = cls()
        n, s = max(bounds[0], bounds[2]), min(bounds[0], bounds[2])
        w, e = min(bounds[1], bounds[3]), max(bounds[1], bounds[3])
        result.nwse = n, w, s, e
        cls.cache[bounds] = result
        return result

    @classmethod
    def from_centroid(cls, centroid: tuple[float, float] | list[float]) -> Self:
        centroid = tuple(centroid)
        if centroid in cls.cache:
            return cls.cache[centroid]
        result = cls()
        result.centroid = centroid
        cls.cache[centroid] = result
        return result

    @classmethod
    def from_geometry(
            cls,
            geometry: str
    ) -> Self:
        result = cls()
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
        return result

    @classmethod
    def from_frame(cls, frame: GeoDataFrame | str | Path) -> Self:
        # if isinstance(frame, (str, Path)):
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
            raise ValueError(
                f"Could not infer frame from '{frame}'"
            )
        result = cls()
        result.geometry = frame
        return result

    @cached_property
    def nwse(self) -> tuple[float, ...]:
        logger.info(
            f"Geocoding {self.address}, this may take a while..."
        )
        # noinspection PyTypeChecker
        nom = None
        try:
            nom = (
                Nominatim(user_agent='tile2net')
                .geocode(self.address, timeout=None)
            )
        except Exception as e:
            logger.debug(f"Nominatim geocoding failed: {e}, trying Photon")

        if nom is not None:
            logger.info(
                f"Geocoded '{self.address}' to\n\t"
                f"'{nom.raw['display_name']}'"
            )
            raw = map(float, nom.raw['boundingbox'])
            s, n, w, e = map(self._round, raw)
            return n, w, s, e

        # noinspection PyTypeChecker
        photon = (
            Photon(user_agent='tile2net')
            .geocode(self.address, timeout=None)
        )
        if photon is None:
            raise ValueError(f"Could not geocode '{self.address}'")
        logger.info(
            f"Geocoded '{self.address}' to\n\t"
            f"'{photon.raw['properties'].get('name', self.address)}'"
        )
        extent = photon.raw['properties'].get('extent')
        if extent is None:
            # point-type Photon results (e.g. individual POIs) have no
            # extent; fall back to a small box around the point
            lon = photon.raw['geometry']['coordinates'][0]
            lat = photon.raw['geometry']['coordinates'][1]
            extent = [lon - 1e-4, lat + 1e-4, lon + 1e-4, lat - 1e-4]
        w, n, e, s = map(self._round, extent)
        return n, w, s, e

    @cached_property
    def wsen(self):
        n, w, s, e = self.nwse
        return w, s, e, n

    @cached_property
    def address(self) -> str:
        # noinspection PyTypeChecker
        reverse = None
        try:
            reverse = (
                Nominatim(user_agent='tile2net')
                .reverse(self.centroid, timeout=None)
            )
        except Exception as e:
            logger.debug(f"Nominatim reverse geocoding failed: {e}, trying Photon")

        if reverse is not None:
            result = reverse.raw['display_name']
            return result

        # noinspection PyTypeChecker
        reverse = (
            Photon(user_agent='tile2net')
            .reverse(self.centroid, timeout=None)
        )
        if reverse is None:
            raise ValueError(f"Could not geocode '{self.centroid}'")
        result = reverse.address
        return result

    @cached_property
    def centroid(self) -> tuple[float, float]:
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

if __name__ == '__main__':
    # GeoCode.from_geometry('New York City')
    # GeoCode.from_geometry('Washington Square Park, New York, NY, USA')
    # GeoCode.from_geometry('40.70661915280362, -74.01066228152449')
    # GeoCode.from_geometry('55 Exchange Pl #5, New York, NY 10005')
    GeoCode.from_osm('New York City')
    GeoCode.from_osm('Washington Square Park, New York, NY, USA')
    GeoCode.from_osm('40.70661915280362, -74.01066228152449')
    GeoCode.from_osm('55 Exchange Pl #5, New York, NY 10005')
