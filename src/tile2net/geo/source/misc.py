from __future__ import annotations

from functools import cached_property

from geopandas import GeoSeries
from shapely import box

from tile2net.geo.source.remote import Remote


class AlamedaCounty(Remote):
    """Alameda County, California - Pictometry WMTS service."""
    # https://www.arcgis.com/home/item.html?id=46db377005dc4e76bbc234c680771573

    name = 'al'
    extension = 'png'
    keyword = dict(
        state=('California', 'CA'),
        # county=('Alameda', 'AL' ),
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
        data = [box(-122.345118999, 37.451422681, -121.462793698, 37.912241999)]
        out = GeoSeries(data, crs='EPSG:4326')
        return out

