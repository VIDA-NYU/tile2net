from __future__ import annotations

from abc import ABC
from functools import cached_property

from geopandas import GeoSeries
from shapely import box

from tile2net.grid.source.remote import Remote


class AlamedaCounty(Remote):
    """Alameda County, California - Pictometry WMTS service."""
    # https://www.arcgis.com/home/item.html?id=46db377005dc4e76bbc234c680771573

    name = 'al'
    extension = 'png'
    keyword = dict(
        state=('California', 'CA'),
        county='Alameda',
    )
    year = 2023
    zoom = 20

    @cached_property
    def server(self) -> str:
        return (
            'https://svc.pictometry.com/Image/'
            '6D9E15C5-C6B4-4ACB-A244-4C44ECA33D90/'
            'wmts/PICT-CAALAM20-OQ74EOAkBw/default/GoogleMapsCompatible'
        )

    @cached_property
    def template(self) -> str:
        return self.server + '/{z}/{x}/{y}.png'

    @cached_property
    def scheme(self) -> str:
        return 'https'

    @cached_property
    def netloc(self) -> str:
        return 'svc.pictometry.com'

    @cached_property
    def path(self) -> str:
        return (
            '/Image/6D9E15C5-C6B4-4ACB-A244-4C44ECA33D90/'
            'wmts/PICT-CAALAM20-OQ74EOAkBw/default/GoogleMapsCompatible/{z}/{x}/{y}.png'
        )

    @cached_property
    def original(self) -> str:
        return self.template

    @cached_property
    def coverage(self) -> GeoSeries:
        return GeoSeries([
            box(
                -122.345118999,  # xmin (W)
                37.451422681,  # ymin (S)
                -121.462793698,  # xmax (E)
                37.912241999  # ymax (N)
            )
        ], crs='EPSG:4326')


class SanFranciscoBase(Remote, ABC):
    """
    Base class for San Francisco aerial imagery.

    San Francisco provides historical aerial imagery dating back to 2014
    via their tile server at tile.sf.gov.
    """

    extension = "png"
    keyword = dict(
        state=('California', 'CA'),
        city='San Francisco',
    )
    zoom = 20
    dimension = 512  # tile size in px

    @cached_property
    def enabled(self) -> bool:
        # Most SF years are outdated, overridden in current year
        return False

    @cached_property
    def template(self) -> str:
        return f"{self.server}/{{z}}/{{x}}/{{y}}.png"

    @cached_property
    def scheme(self) -> str:
        return 'https'

    @cached_property
    def netloc(self) -> str:
        return 'tile.sf.gov'

    @cached_property
    def path(self) -> str:
        # Extract path from server
        server_path = self.server.replace('https://tile.sf.gov', '')
        return f"{server_path}/{{z}}/{{x}}/{{y}}.png"

    @cached_property
    def original(self) -> str:
        return self.template

    @cached_property
    def coverage(self) -> GeoSeries:
        bounds = box(-122.514926, 37.708075, -122.356779, 37.832371)
        return GeoSeries([bounds], crs="EPSG:4326")


class SanFrancisco2014(SanFranciscoBase):
    name = 'sf2014'
    year = 2014
    server = 'https://tile.sf.gov/api/tiles/p2014_rgb8cm'


class SanFrancisco2017(SanFranciscoBase):
    name = 'sf2017'
    year = 2017
    server = 'https://tile.sf.gov/api/tiles/p2017_rgb8cm'


class SanFrancisco2018(SanFranciscoBase):
    name = 'sf2018'
    year = 2018
    server = 'https://tile.sf.gov/api/tiles/p2018_rgb8cm'


class SanFrancisco2019(SanFranciscoBase):
    name = 'sf2019'
    year = 2019
    server = 'https://tile.sf.gov/api/tiles/p2019_rgb8cm'


class SanFrancisco2020(SanFranciscoBase):
    name = 'sf2020'
    year = 2020
    server = 'https://tile.sf.gov/api/tiles/p2020_rgb8cm'


class SanFrancisco2023(SanFranciscoBase):
    name = 'sf2023'
    year = 2023
    server = 'https://tile.sf.gov/api/tiles/p2023_rgb8cm'


class SanFrancisco2024(SanFranciscoBase):
    name = 'sf2024'
    year = 2024
    server = 'https://tile.sf.gov/api/tiles/p2024_rgb8cm'

    @cached_property
    def enabled(self) -> bool:
        return True  # Current year is enabled
