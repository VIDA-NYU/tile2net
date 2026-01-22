from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Self

import geopandas as gpd
import requests
import shapely.geometry
from geopandas import GeoSeries
from requests.adapters import HTTPAdapter
from shapely import box, wkt
from urllib3.util.retry import Retry

from tile2net.grid.source.exceptions import SourceParseError
from tile2net.grid.source.remote import Remote


class ArcGis(
    Remote,
    ABC
):
    """Base class for ArcGIS tile servers."""

    server: str
    """Base URL for the ArcGIS MapServer."""

    @cached_property
    def response(self) -> dict:
        params = {'f': 'json'}
        response = requests.get(self.server, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data

    @cached_property
    def coverage(self) -> gpd.GeoSeries:
        """
        Returns the spatial extent of the ArcGIS service as a GeoSeries.
        The method extracts the bounding box from the 'fullExtent' metadata
        and projects it to EPSG:4326.
        """
        data = self.response
        extent = data.get('fullExtent') or data.get('initialExtent')
        if not extent:
            return gpd.GeoSeries()
        crs = f'EPSG:{extent["spatialReference"].get("latestWkid", 3857)}'
        geometry = box(extent['xmin'], extent['ymin'], extent['xmax'], extent['ymax'])
        out = (
            gpd.GeoSeries([geometry], crs=crs)
            .to_crs(epsg=4326)
        )
        return out

    @cached_property
    def zooms(self) -> list[int]:
        """Fetches metadata from the ArcGIS REST API and extracts supported zoom levels."""
        return [
            int(lod.get('level'))
            for lod in self.response['tileInfo']['lods']
        ]

    @cached_property
    def zoom(self):
        if not self.zooms:
            out = 20
        else:
            out = max(self.zooms)
            out = min(out, 20)
        return out

    @cached_property
    def dimension(self) -> int:
        rows = self.response['tileInfo']['rows']
        cols = self.response['tileInfo']['cols']
        assert rows == cols, "Non-square tiles are not supported"
        return rows

    @cached_property
    def template(self) -> str:
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
        if '/tile/' in value:
            value = value.split('/tile/')[0]
        out.server = value

        return out

    @classmethod
    def from_server(
            cls,
            value: str,
            name: str | None = None,
    ) -> Self:
        """Create ArcGis instance from a server URL with validation."""

        url = value.strip().rstrip('/')
        if '/tile/' in url:
            url = url.split('/tile/')[0]

        # Configure robust retry strategy for the validation request
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)

        try:
            response = session.get(
                url,
                params={'f': 'json'},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            msg = f"Failed to connect or parse ArcGIS response from {url}: {e}"
            raise SourceParseError(msg) from e

        if 'error' in data:
            msg = f"ArcGIS server returned error: {data['error']}"
            raise SourceParseError(msg)

        # Validate content using multi-line assignment for clarity
        has_version = 'currentVersion' in data
        has_services_meta = 'spatialReference' in data
        has_services_meta &= 'layers' in data or 'initialExtent' in data
        has_catalog_meta = 'services' in data or 'folders' in data

        is_valid = has_version
        is_valid &= has_services_meta or has_catalog_meta
        if not is_valid:
            msg = f"URL does not appear to be a valid ArcGIS MapServer: {value}"
            raise SourceParseError(msg)

        out = cls()
        out.server = url
        out.response = data

        return out

# These classes are now provided in the `servers.yaml` but left here with `enabled=False` for use as examples.

class NewYorkCity(ArcGis):
    enabled = False
    server = 'https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer'
    name = 'nyc'
    keyword = dict(
        state=('New York', 'NY'),
        city=('New York', 'NYC'),
    )
    year = 2024
    coverage = GeoSeries(
        wkt.loads(
            "POLYGON ((-73.69048 40.4889, -73.69048 40.92834, -74.27615 40.92834, -74.27615 40.4889, -73.69048 40.4889))"),
        crs='epsg:4326'
    )


class NewYork(ArcGis):
    enabled = False
    server = 'https://orthos.its.ny.gov/arcgis/rest/services/wms/2024/MapServer'
    name = 'ny'
    keyword = dict(
        state=('New York', 'NY'),
    )
    dropword = dict(
        city='New York',
    )
    year = 2024
    coverage = GeoSeries(
        wkt.loads(
            "POLYGON ((-73.25509 40.48341, -73.25509 45.03931, -79.77922 45.03931, -79.77922 40.48341, -73.25509 40.48341))"),
        crs='epsg:4326'
    )


class Massachusetts(ArcGis):
    enabled = False
    server = 'https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/orthos2021/MapServer'
    name = 'ma'
    keyword = dict(
        state=('Massachusetts', 'MA'),
    )
    extension = 'jpg'
    year = 2021
    coverage = GeoSeries(
        wkt.loads(
            "POLYGON ((-69.92466 41.2265, -69.92466 42.89199, -73.51979 42.89199, -73.51979 41.2265, -69.92466 41.2265))"),
        crs='epsg:4326'
    )


class KingCountyWashington(ArcGis):
    enabled = False
    server = 'https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2023/MapServer'
    name = 'king'
    keyword = dict(
        state=('Washington', 'WA'),
        county='King',
    )
    year = 2023
    coverage = GeoSeries(
        wkt.loads(
            "POLYGON ((-121.03559 47.04875, -121.03559 47.96618, -122.56958 47.96618, -122.56958 47.04875, -121.03559 47.04875))"),
        crs='epsg:4326'
    )


class LosAngeles(ArcGis):
    enabled = False
    server = 'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Aerial_2014/MapServer'
    name = 'la'
    keyword = dict(
        county='Los Angeles',
    )
    year = 2014
    coverage = GeoSeries(
        wkt.loads(
            "POLYGON ((-117.63102 33.28853, -117.63102 34.83012, -118.95937 34.83012, -118.95937 33.28853, -117.63102 33.28853))"),
        crs='epsg:4326'
    )


class NewJersey(ArcGis):
    enabled = False
    server = 'https://maps.nj.gov/arcgis/rest/services/Basemap/Orthos_Natural_2020_NJ_WM/MapServer'
    name = 'nj'
    keyword = dict(
        state=('New Jersey', 'NJ'),
    )
    year = 2020
    coverage = GeoSeries(
        wkt.loads(
            "POLYGON ((-73.85543 38.824, -73.85543 41.3866, -75.59981 41.3866, -75.59981 38.824, -73.85543 38.824))"),
        crs='epsg:4326'
    )


class SpringHillTN(ArcGis):
    enabled = False
    server = 'https://tiles.arcgis.com/tiles/tF0XsRR9ptiKNVW2/arcgis/rest/services/Spring_Hill_Imagery_WGS84/MapServer'
    name = 'sh_tn'
    keyword = dict(
        state=('Tennessee', 'TN'),
        city='Spring Hill',
        town='Spring Hill',
    )
    year = 2020
    coverage = GeoSeries(
        wkt.loads(
            "POLYGON ((-86.75604 35.58884, -86.75604 35.85806, -87.13095 35.85806, -87.13095 35.58884, -86.75604 35.58884))"),
        crs='epsg:4326'
    )


class Virginia(ArcGis):
    enabled = False
    """Data from https://vgin.vdem.virginia.gov/pages/orthoimagery"""
    server = "https://vginmaps.vdem.virginia.gov/arcgis/rest/services/VBMP_Imagery/MostRecentImagery_WGS/MapServer/"
    name = "va"
    keyword = dict(
        state=('Virginia', 'VA'),
    )
    coverage = GeoSeries(
        shapely.geometry.box(-83.6753, 36.5407, -75.1664, 39.4660),
        crs='epsg:4326'
    )


class MaineOrthoBase(ArcGis, ABC):
    """
    Shared config for Maine GeoLibrary statewide imagery.
    (Everything else—coverage, zoom, template—comes from ArcGis.)
    """
    keyword = dict(
        state=('Maine', 'ME'),
    )
    extension = "jpeg"
