
from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC
from functools import cached_property
from typing import Iterable
from urllib.parse import urlencode

import requests
from geopandas import GeoSeries
from requests.adapters import HTTPAdapter
from shapely import wkt
from urllib3.util.retry import Retry

from .remote import Remote


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
        """Original URL template for the WMTS tiles."""
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
