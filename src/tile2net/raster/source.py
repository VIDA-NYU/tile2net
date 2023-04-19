from __future__ import annotations

import geopandas as gpd
import pyproj

import pandas as pd
import toolz
from shapely import transform
from toolz import pipe, curried, curry
from toolz import pipe, curried, curry
from abc import ABCMeta, ABC, abstractmethod
from functools import cached_property
import shapely.ops
import shapely.geometry

import numpy as np
import shapely.geometry
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

from typing import Iterable, Iterator, Optional, Type, Union

from tile2net.misc.frame import Frame


if False:
    from tile2net.raster.raster import Raster
    from tile2net.raster.tile import Tile

class SourceMeta(ABCMeta):
    @classmethod
    @property
    def coverage(cls) -> GeoSeries:
        coverages: list[GeoSeries] = [
            source.coverage
            .set_crs('epsg:4326')
            .set_axis(
                pd.Index([source.name] * len(source.coverage), name='source'),
            )
            for source in cls.catalog.values()
        ]
        coverage = pd.concat(coverages)
        return coverage

    def __getitem__(
        cls,
        item: list[float] | str | shapely.geometry.base.BaseGeometry,
    ) -> Optional['Source']:

        if isinstance(item, list):
            s, w, n, e = item
            item = shapely.geometry.box(w, s, e, n)

        if isinstance(item, shapely.geometry.base.BaseGeometry):
            trans = pyproj.Transformer.from_crs(
                'epsg:4326', 'epsg:3857', always_xy=True
            ).transform
            item = shapely.ops.transform(trans, item)

            matches: GeoSeries = (
                cls.__class__.coverage.geometry
                # SourceMeta.coverage.geometry

                .to_crs('epsg:3857')
                .loc[lambda x: x.intersects(item)]
            )
            if matches.empty:
                raise KeyError(f'No source found for {item}')
            item = (
                matches.intersection(item)
                .area
                .__rtruediv__(matches.area)
                .idxmax()
            )

        if isinstance(item, str):
            if item not in cls.catalog:
                raise KeyError(f'No source found for {item}')
            source = cls.catalog[item]

        else:
            raise TypeError(f'Invalid type {type(item)} for {item}')
        return source()

    def __init__(self: Type[Source], name, bases, attrs):
        super().__init__(name, bases, attrs)
        if self.coverage is not None:
            if self.name is None:
                raise ValueError(f'{self} must have a name')
            if self.name in self.catalog:
                raise ValueError(f'{self} name already in use')
            self.catalog[self.name] = self

    catalog: dict[str, Type['Source']] = {}

class Source(ABC, metaclass=SourceMeta):
    name: str = None
    coverage: GeoSeries = None
    zoom = 19
    extension = 'png'

    def __getitem__(self, item: Iterator[Tile]) -> Iterable[str]:
        ...

    def __bool__(self):
        return True

    def __repr__(self):
        return self.name

class NewYorkCity(Source):
    name = 'nyc'
    zoom = 20
    coverage = GeoSeries([
        shapely.geometry.box(-74.25559, 40.49612, -73.70001, 40.91553),
    ], crs='epsg:4326')

    def __getitem__(self, item: Iterator[Tile]):
        yield from (
            f"https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_-_2020/MapServer" \
            f"/tile/{tile.zoom}/{tile.ytile}/{tile.xtile}"
            for tile in item
        )

class NewYork(Source):
    name = 'ny'
    zoom = 19
    coverage = GeoSeries([
        shapely.geometry.box(-79.762, 40.49612, -71.856, 45.015),
    ], crs='epsg:4326')

    def __getitem__(self, item: Iterator[Tile]):
        yield from (
            f"https://orthos.its.ny.gov/arcgis/rest/services/wms/2020/MapServer" \
            f"/tile/{tile.zoom}/{tile.ytile}/{tile.xtile}"
            for tile in item
        )

class Massachusetts(Source):
    name = 'ma'
    zoom = 20

    coverage = GeoSeries([
        shapely.geometry.box(-73.508, 41.237, -69.928, 42.886),
    ], crs='epsg:4326')

    def __getitem__(self, item: Iterator[Tile]):
        yield from (
            f"https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/USGS_Orthos_2019/MapServer" \
            f"/tile/{tile.zoom}/{tile.ytile}/{tile.xtile}"
            for tile in item
        )

class KingCountyWashington(Source):
    name = 'king'

    coverage = GeoSeries([
        shapely.geometry.box(-122.454, 47.409, -121.747, 47.734),
    ], crs='epsg:4326')

    def __getitem__(self, item: Iterator[Tile]):
        yield from (
            f"https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/Kings_County_Orthos_-_2020/MapServer" \
            f"/tile/{tile.zoom}/{tile.ytile}/{tile.xtile}"
            for tile in item
        )

class WashingtonDC(Source):
    name = 'dc'

    coverage = GeoSeries([
        shapely.geometry.box(-77.119, 38.791, -76.909, 38.995),
    ], crs='epsg:4326')

    def __getitem__(self, item: Iterator[Tile]):
        for tile in item:
            top, left, bottom, right = tile.transformProject(tile.crs, 3857)
            yield (
                f'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer'
                f'/exportImage?f=image&bbox={bottom}%2C{right}%2C{top}%2C{left}'
                f'&imageSR=102100&bboxSR=102100&size=512%2C512'
            )

class LosAngeles(Source):
    name = 'la'

    coverage = GeoSeries([
        shapely.geometry.box(-118.668, 33.703, -118.155, 34.337),
    ], crs='epsg:4326')

    def __getitem__(self, item: Iterator[Tile]):
        yield from (
            f'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Aerial_2014/MapServer'
            f'/tile/{tile.zoom}/{tile.xtile}/{tile.ytile}'
            for tile in item
        )

class WestOregon(Source):
    name = 'w_or'
    extension = 'jpeg'
    coverage = GeoSeries([
        shapely.geometry.box(
            -1.3873498889181776E7,
            5156074.079470176,
            -1.3316477481920356E7,
            5835830.064712983,
        ),
    ], crs=3857).to_crs(4326)

    def __getitem__(self, item: Iterator[Tile]):
        for tile in item:
            yield (
                f'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2018/OSIP_2018_WM/ImageServer'
                f'/tile/{tile.zoom}/{tile.ytile}/{tile.xtile}'
            )

class EastOregon(Source):
    name = 'e_or'
    extension = 'jpeg'
    coverage = GeoSeries([
        shapely.geometry.box(
            -1.350803972228365E7,
            5156085.127009435,
            -1.2961447490647841E7,
            5785584.363334397,
        ),
    ], crs=3857).to_crs(4326)

    def __getitem__(self, item: Iterator[Tile]):
        """https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2017/OSIP_2017_WM/ImageServer/tile/19/186453/85422"""
        for tile in item:
            yield (
                f'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2017/OSIP_2017_WM/ImageServer'
                f'/tile/{tile.zoom}/{tile.ytile}/{tile.xtile}'
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

