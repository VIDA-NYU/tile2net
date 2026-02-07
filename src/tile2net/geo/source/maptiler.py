from __future__ import annotations

from abc import ABC
from functools import cached_property

import requests
from geopandas import GeoSeries
from shapely import box

from tile2net.geo.source.remote import Remote


class MapTiler(
    Remote,
    ABC,
    base=True,
):
    """
    Base class for MapTiler-based tile servers.
    Dynamically fetches metadata from the /tiles.json endpoint.

    https://docs.maptiler.com/server/api/tiles/
    """

    server: str
    """Base URL for the MapTiler server (e.g. https://tile.sf.gov/api/tiles/p2024_rgb8cm)."""

    @cached_property
    def response(self) -> dict:
        """
        Fetches metadata from the MapTiler 'tiles.json' endpoint.
        """
        # MapTiler Server standard: metadata is at /tiles.json
        url = f"{self.server.rstrip('/')}/tiles.json"

        # Use a session for connection pooling/retries if needed,
        # strictly following the user's preference for simple requests first
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    @cached_property
    def coverage(self) -> GeoSeries:
        """
        Returns the spatial extent from the metadata.
        Format: [minx, miny, maxx, maxy]
        """
        bounds = self.response.get('bounds')
        if not bounds:
            return GeoSeries()

        # Metadata bounds are already in WGS84 (EPSG:4326)
        geometry = box(bounds[0], bounds[1], bounds[2], bounds[3])
        return GeoSeries([geometry], crs="EPSG:4326")

    @cached_property
    def zoom(self) -> int:
        """Returns the maximum supported zoom level from metadata. """
        out = int(self.response['maxzoom'])
        out = min(out, 20)
        return out

    @cached_property
    def dimension(self) -> int:
        """
        Calculates tile dimension based on the 'scale' property.
        scale: 1 -> 256px
        scale: 2 -> 512px
        """
        scale = int(self.response.get('scale', 1))
        return 256 * scale

    @cached_property
    def template(self) -> str:
        """
        Returns the XYZ template.
        MapTiler metadata usually provides a 'tiles' list with templates,
        but constructing it ensures consistency with our class structure.
        """
        return f"{self.server}/{{z}}/{{x}}/{{y}}.{self.extension}"

    @cached_property
    def scheme(self) -> str:
        return 'https'

    @cached_property
    def extension(self) -> str:
        """
        Returns the file extension from metadata 'format' (e.g. 'png', 'jpg').
        """
        return self.response.get('format', 'png')

    @cached_property
    def path(self) -> str:
        """
        Extracts the path component for the tile request.
        """
        # Remove the scheme and domain to get the base path
        # e.g. https://tile.sf.gov/api/tiles/p2024 -> /api/tiles/p2024
        parts = self.server.split('://', 1)[-1].split('/', 1)
        base_path = f"/{parts[1]}" if len(parts) > 1 else ""
        return f"{base_path}/{{z}}/{{x}}/{{y}}.{self.extension}"

    @cached_property
    def original(self) -> str:
        return self.server


# https://tile.sf.gov

class SanFrancisco2014(MapTiler):
    enabled = False
    name = 'sf2014'
    year = 2014
    server = 'https://tile.sf.gov/api/tiles/p2014_rgb8cm'
    keyword = dict(state=('California', 'CA'))


class SanFrancisco2017(MapTiler):
    enabled = False
    name = 'sf2017'
    year = 2017
    server = 'https://tile.sf.gov/api/tiles/p2017_rgb8cm'
    keyword = dict(state=('California', 'CA'))


class SanFrancisco2018(MapTiler):
    enabled = False
    name = 'sf2018'
    year = 2018
    server = 'https://tile.sf.gov/api/tiles/p2018_rgb8cm'
    keyword = dict(state=('California', 'CA'))


class SanFrancisco2019(MapTiler):
    enabled = False
    name = 'sf2019'
    year = 2019
    server = 'https://tile.sf.gov/api/tiles/p2019_rgb8cm'
    keyword = dict(state=('California', 'CA'))


class SanFrancisco2020(MapTiler):
    enabled = False
    name = 'sf2020'
    year = 2020
    server = 'https://tile.sf.gov/api/tiles/p2020_rgb8cm'
    keyword = dict(state=('California', 'CA'))


class SanFrancisco2023(MapTiler):
    enabled = False
    name = 'sf2023'
    year = 2023
    server = 'https://tile.sf.gov/api/tiles/p2023_rgb8cm'
    keyword = dict(state=('California', 'CA'))


class SanFrancisco2024(MapTiler):
    enabled = True
    name = 'sf2024'
    year = 2024
    server = 'https://tile.sf.gov/api/tiles/p2024_rgb8cm'
    keyword = dict(state=('California', 'CA'))
