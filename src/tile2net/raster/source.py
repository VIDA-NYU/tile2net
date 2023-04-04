from abc import ABCMeta, ABC, abstractmethod
from functools import cached_property

import numpy as np
import shapely.geometry
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

from typing import Iterable, Iterator, Optional, Type


if False:
    from tile2net.raster.raster import Raster
    from tile2net.raster.tile import Tile

class SourceMeta(ABCMeta):
    # def __contains__(self: Type['Source'], item):
    #     return item in self.coverage

    def __getitem__(self, item: list[float]) -> Optional['Source']:
        s, w, n, e = item
        bbox = shapely.geometry.box(w, s, e, n)
        for source in self._lookup:
            if (
                    source.coverage.geometry
                            .intersects(bbox)
                            .any()
            ):
                return source()
        raise KeyError(f'No source found for {item}')

    _lookup: list[Type['Source']] = []

class Source(ABC, metaclass=SourceMeta):
    name: str
    coverage: GeoSeries
    zoom = 19

    def __getitem__(self, item: np.ndarray) -> Iterable[str]:
        ...

    def __init_subclass__(cls, **kwargs):
        cls._lookup.append(cls)
        super().__init_subclass__(**kwargs)

class NewYorkCity(Source):
    name = 'nyc'
    zoom = 20
    coverage = GeoSeries([
        shapely.geometry.box(-74.25559, 40.49612, -73.70001, 40.91553),
    ], crs='epsg:4326')

    def __getitem__(self, item: np.ndarray):
        yield from (
            f"https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/"
            f"NYC_Orthos_-_2020/MapServer/tile/{tile.zoom}/{tile.ytile}/{tile.xtile}"
            for tile in item.flat
        )

class NewYork(Source):
    name = 'ny'
    zoom = 20
    coverage = GeoSeries([
        shapely.geometry.box(-79.762, 40.49612, -71.856, 45.015),
    ], crs='epsg:4326')

    def __getitem__(self, item: np.ndarray):
        yield from (
            f"https://orthos.its.ny.gov/arcgis/rest/services/wms/2020/MapServer/tile/"
            f"{tile.zoom}/{tile.ytile}/{tile.xtile}"
            for tile in item.flat
        )

class Massachusetts(Source):
    name = 'ma'
    zoom = 20

    coverage = GeoSeries([
        shapely.geometry.box(-73.508, 41.237, -69.928, 42.886),
    ], crs='epsg:4326')

    def __getitem__(self, item: np.ndarray):
        yield from (
            f"https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/"
            f"USGS_Orthos_2019/"
            f"MapServer/tile/{tile.zoom}/{tile.ytile}/{tile.xtile}"
            for tile in item.flat
        )

class KingCountyWashington(Source):
    name = 'king'

    coverage = GeoSeries([
        shapely.geometry.box(-122.454, 47.409, -121.747, 47.734),
    ], crs='epsg:4326')

    def __getitem__(self, item: np.ndarray):
        yield from (
            f"https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/"
            f"Kings_County_Orthos_-_2020/MapServer/tile/{tile.zoom}/{tile.ytile}/{tile.xtile}"
            for tile in item.flat
        )

class WashingtonDC(Source):
    name = 'dc'

    coverage = GeoSeries([
        shapely.geometry.box(-77.119, 38.791, -76.909, 38.995),
    ], crs='epsg:4326')

    def __getitem__(self, item: np.ndarray):
        for tile in item.flat:
            top, left, bottom, right = tile.transformProject(tile.crs, 3857)
            yield (
                f'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/'
                f'ImageServer'
                f'/exportImage?f=image&bbox={bottom}%2C{right}%2C{top}%2C{left}'
                f'&imageSR=102100&bboxSR=102100&size=512%2C512'
            )

class LosAngeles(Source):
    name = 'la'

    coverage = GeoSeries([
        shapely.geometry.box(-118.668, 33.703, -118.155, 34.337),
    ], crs='epsg:4326')

    def __getitem__(self, item: np.ndarray):
        yield from (
            f'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/'
            f'LACounty_Aerial_2014/'
            f'MapServer/tile/{tile.zoom}/{tile.xtile}/{tile.ytile}'
            for tile in item.flat
        )


if __name__ == '__main__':
    lat1, lon1 = 40.59477460446395, -73.96014473965148
    lat2, lon2 = 40.636082070035755, -73.92478249851513
    coverage = [
        min(lat1, lat2),
        max(lat1, lat2),
        min(lon1, lon2),
        max(lon1, lon2),
    ]
    source = Source[coverage]
    print(cls)
