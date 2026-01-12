from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC
from functools import cached_property
from typing import Iterable
from urllib.parse import urlencode

import requests
from geopandas import GeoSeries
from requests.adapters import HTTPAdapter
from shapely import box, wkt
from urllib3.util.retry import Retry

from .remote import Remote



class AlamedaCounty(Remote):
    """Alameda County, California - Pictometry WMTS service."""
    # https://www.arcgis.com/home/item.html?id=46db377005dc4e76bbc234c680771573

    name = 'al'
    extension = 'png'
    keyword = 'Alameda County', 'California'
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
    def format(self) -> str:
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
        return self.format

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
    keyword = "San Francisco", "California"
    zoom = 20
    dimension = 512  # tile size in px

    @cached_property
    def enabled(self) -> bool:
        # Most SF years are outdated, overridden in current year
        return False

    @cached_property
    def format(self) -> str:
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
        return self.format

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


class VexCel(
    Remote,
    ABC
):
    """
    Base class for VexCel WMTS imagery services.

    VexCel provides high-resolution aerial imagery through a WMTS API.
    Subclasses must define api_key and keyword attributes.
    """

    flip_y = False
    prefer_layers: Iterable[str] = ('wide-area', 'urban', 'urban-r', 'graysky')
    base = 'https://api.vexcelgroup.com/v2/ortho/wmts'
    timeout = 10
    extension = 'png'

    @cached_property
    def server(self) -> str:
        return 'https://api.vexcelgroup.com/v2/ortho'

    @cached_property
    def scheme(self) -> str:
        return 'https'

    @cached_property
    def netloc(self) -> str:
        return 'api.vexcelgroup.com'

    @cached_property
    def api_key(self) -> str:
        """API key for VexCel service. Must be defined in subclasses."""
        raise AttributeError("api_key must be defined in subclass")

    @cached_property
    def _session(self) -> requests.Session:
        """Create a requests session with retries and a custom User-Agent."""
        s = requests.Session()
        s.headers.update({'User-Agent': 'tile2net'})
        retry = Retry(
            total=5,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        s.mount('https://', HTTPAdapter(max_retries=retry))
        s.mount('http://', HTTPAdapter(max_retries=retry))
        return s

    @classmethod
    def _raise_http(
            cls,
            r: requests.Response,
            url: str,
    ) -> None:
        """Raise an HTTPError with details from the response."""
        ctype = r.headers.get('Content-Type', '')
        if ctype.startswith('text/'):
            body = r.text[:800]
        else:
            body = f'<{len(r.content):,} bytes binary>'
        raise requests.HTTPError(
            f"{r.status_code} {r.reason or ''} for {url}\n"
            f"CT: {ctype}\n"
            f"Body: {body}"
        )

    @cached_property
    def layers(self) -> list[str]:
        """
        Layers available in the WMTS service.
        e.g. ['wide-area', 'urban', 'urban-r', 'graysky']
        """
        query = dict(
            SERVICE='WMTS',
            REQUEST='GetCapabilities',
            VERSION='1.0.0',
            api_key=self.api_key,
        )
        url = f"{self.base}?{urlencode(query)}"

        with self._session as s:
            r = s.get(url, timeout=self.timeout)
            if r.status_code >= 400:
                self._raise_http(r, url)
            try:
                root = ET.fromstring(r.content)
            except ET.ParseError as e:
                raise RuntimeError(f'Failed to parse WMTS capabilities: {e}') from e

        ns = dict(
            wmts='http://www.opengis.net/wmts/1.0',
            ows='http://www.opengis.net/ows/1.1',
        )
        result = []
        for lyr in root.findall('.//wmts:Contents/wmts:Layer', ns):
            ident = lyr.find('ows:Identifier', ns)
            if ident is not None and ident.text:
                result.append(ident.text.strip())
        return result

    @cached_property
    def layer(self) -> str:
        """Layer chosen from the WMTS layers."""
        avail = {
            layer
            for layer in self.layers
            if not layer.lower().endswith('-g')  # exclude greyscale imagery
        }
        for cand in self.prefer_layers:
            if cand in avail:
                return cand
        if not avail:
            raise RuntimeError('No RGB WMTS layers are available for this API key.')
        result = sorted(avail)[0]
        return result

    @cached_property
    def format(self) -> str:
        """Encoding the WMTS URL template for the tiles."""
        query = {
            'layer': self.layer,
            'image-format': self.extension,
            'api_key': self.api_key,
        }
        q = urlencode(query)
        q += '&tile-x={x}&tile-y={y}&zoom={z}'
        result = f'{self.server}/tile?{q}'
        return result

    @cached_property
    def path(self) -> str:
        """URL path component for WMTS tiles."""
        query = {
            'layer': self.layer,
            'image-format': self.extension,
            'api_key': self.api_key,
        }
        q = urlencode(query)
        q += '&tile-x={x}&tile-y={y}&zoom={z}'
        return f'/v2/ortho/tile?{q}'

    @cached_property
    def original(self) -> str:
        return self.format


class Maine(VexCel):
    """Maine statewide aerial imagery via VexCel WMTS."""
    # https://www.maine.gov/geolib/data/index.html

    name = 'maine'
    api_key = "vfa_JkRqdw7HHOis.37zZe0OHVVqPlIY91FsT9arC9uGrDalMqwMW1AX4OgcfsiwtQoxDLed9OEIxKy3Ys2lMCam9C2swLrUwNqX2KrlegBRev8MRDpkqHkbSEn0fP1aEqvoDBdePjAOO9h91.4256792737"
    keyword = 'Maine'

    @cached_property
    def coverage(self) -> GeoSeries:
        return GeoSeries(
            wkt.loads(
                "MULTIPOLYGON (((-70.64573401557249 43.09008331966716, "
                "-70.75102474636725 43.08003225358636, "
                "-70.79761105007827 43.21973948828747, "
                "-70.98176001655037 43.36789581966831, "
                "-70.94416541205806 43.46633942318429, "
                "-71.08481999999998 45.30523999999996, "
                "-70.6600225491012 45.460222886733995, "
                "-70.30577485483874 45.91584605942051, "
                "-70.23252499999999 46.67723999999999, "
                "-70.04698876482551 47.27759104512078, "
                "-69.22949648234014 47.45839057788926, "
                "-69.04372176801829 47.412188489118806, "
                "-68.88897897699865 47.18540375261842, "
                "-68.23512724683671 47.35486094843281, "
                "-67.79011605789268 47.06641095953984, "
                "-67.7897945726203 45.93829285562829, "
                "-67.80993374244043 45.70223874247503, "
                "-67.13734239106576 45.13732765323097, "
                "-66.96465667699112 44.8095336106854, "
                "-67.31909616052107 44.68957147935146, "
                "-68.02807343724522 44.35658965174269, "
                "-68.90084652253765 43.900781458273264, "
                "-69.06007042678839 44.09555237311301, "
                "-69.2304350882579 43.85690506766196, "
                "-69.98948999999997 43.46771999999999, "
                "-70.11614164561623 43.68063838486712, "
                "-70.64573401557249 43.09008331966716)))"
            ),
            crs='epsg:4326'
        )