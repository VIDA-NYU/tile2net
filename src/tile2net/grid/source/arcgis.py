from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Self

import shapely.geometry
from geopandas import GeoSeries
from shapely import box, wkt

from .remote import Remote


class ArcGis(
    Remote,
    ABC
):
    """
    Base class for ArcGIS tile servers.
    """

    @cached_property
    def server(self) -> str:
        """Base URL for the ArcGIS MapServer."""
        raise AttributeError()

    @cached_property
    def format(self) -> str:
        """URL template for ArcGIS tile requests."""
        return f"{self.server}/tile/{{z}}/{{y}}/{{x}}"

    @cached_property
    def scheme(self) -> str:
        """Extract scheme from server URL."""
        if self.server.startswith('https://'):
            return 'https '
        elif self.server.startswith('http://'):
            return 'http'
        return 'https'

    @cached_property
    def netloc(self) -> str:
        """Extract network location from server URL."""
        url = self.server
        # Remove scheme
        if '://' in url:
            url = url.split('://', 1)[1]
        # Get domain part
        return url.split('/')[0]

    @cached_property
    def path(self) -> str:
        """Extract path from server URL."""
        url = self.server
        # Remove scheme
        if '://' in url:
            url = url.split('://', 1)[1]
        # Get everything after domain
        parts = url.split('/', 1)
        if len(parts) > 1:
            return '/' + parts[1] + '/tile/{z}/{y}/{x}'
        return '/tile/{z}/{y}/{x}'

    @cached_property
    def original(self) -> str:
        """Original server URL."""
        return self.server

    @cached_property
    def extension(self) -> str:
        """File extension for tiles, default is 'png'."""
        return 'png'

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Create ArcGis instance from a server URL."""
        out = cls()
        value = value.rstrip('/')
        # remove /tile/{z}/{y}/{x} pattern if present
        if '/tile/' in value:
            value = value.split('/tile/')[0]
        out.server = value

        return out



class NewYorkCity(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer'
    name = 'nyc'
    keyword = 'New York City', 'City of New York'
    year = 2024
    coverage = GeoSeries(
        wkt.loads("POLYGON ((-73.69048 40.4889, -73.69048 40.92834, -74.27615 40.92834, -74.27615 40.4889, -73.69048 40.4889))"),
        crs='epsg:4326'
    )


class NewYork(ArcGis):
    server = 'https://orthos.its.ny.gov/arcgis/rest/services/wms/2024/MapServer'
    name = 'ny'
    keyword = 'New York'
    year = 2024
    coverage = GeoSeries(
        wkt.loads("POLYGON ((-73.25509 40.48341, -73.25509 45.03931, -79.77922 45.03931, -79.77922 40.48341, -73.25509 40.48341))"),
        crs='epsg:4326'
    )


class Massachusetts(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/orthos2021/MapServer'
    name = 'ma'
    keyword = 'Massachusetts'
    extension = 'jpg'
    year = 2021
    coverage = GeoSeries(
        wkt.loads("POLYGON ((-69.92466 41.2265, -69.92466 42.89199, -73.51979 42.89199, -73.51979 41.2265, -69.92466 41.2265))"),
        crs='epsg:4326'
    )


class KingCountyWashington(ArcGis):
    server = 'https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2023/MapServer'
    name = 'king'
    keyword = 'King County, Washington', 'King County'
    year = 2023
    coverage = GeoSeries(
        wkt.loads("POLYGON ((-121.03559 47.04875, -121.03559 47.96618, -122.56958 47.96618, -122.56958 47.04875, -121.03559 47.04875))"),
        crs='epsg:4326'
    )


class LosAngeles(ArcGis):
    server = 'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Aerial_2014/MapServer'
    name = 'la'
    keyword = 'Los Angeles'
    year = 2014
    coverage = GeoSeries(
        wkt.loads("POLYGON ((-117.63102 33.28853, -117.63102 34.83012, -118.95937 34.83012, -118.95937 33.28853, -117.63102 33.28853))"),
        crs='epsg:4326'
    )


class NewJersey(ArcGis):
    server = 'https://maps.nj.gov/arcgis/rest/services/Basemap/Orthos_Natural_2020_NJ_WM/MapServer'
    name = 'nj'
    keyword = 'New Jersey'
    year = 2020
    coverage = GeoSeries(
        wkt.loads("POLYGON ((-73.85543 38.824, -73.85543 41.3866, -75.59981 41.3866, -75.59981 38.824, -73.85543 38.824))"),
        crs='epsg:4326'
    )


class SpringHillTN(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/tF0XsRR9ptiKNVW2/arcgis/rest/services/Spring_Hill_Imagery_WGS84/MapServer'
    name = 'sh_tn'
    keyword = 'Spring Hill, Tennessee', 'Spring Hill'
    year = 2020
    coverage = GeoSeries(
        wkt.loads("POLYGON ((-86.75604 35.58884, -86.75604 35.85806, -87.13095 35.85806, -87.13095 35.58884, -86.75604 35.58884))"),
        crs='epsg:4326'
    )


class Virginia(ArcGis):
    """Data from https://vgin.vdem.virginia.gov/pages/orthoimagery"""
    server = "https://gismaps.vdem.virginia.gov/arcgis/rest/services/VBMP_Imagery/MostRecentImagery_WGS/MapServer/"
    name = "va"
    keyword = "Virginia"
    coverage = GeoSeries(
        shapely.geometry.box(-83.6753, 36.5407, -75.1664, 39.4660),
        crs='epsg:4326'
    )


class MaineOrthoBase(ArcGis, ABC):
    """
    Shared config for Maine GeoLibrary statewide imagery.
    (Everything else—coverage, zoom, template—comes from ArcGis.)
    """
    keyword = "Maine"
    extension = "jpeg"
