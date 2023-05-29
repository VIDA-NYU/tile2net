from __future__ import annotations
from toolz import pipe, curried, curry

import functools
import json
from abc import ABC, ABCMeta
from typing import Iterator, Optional, Type
from weakref import WeakKeyDictionary

import pandas as pd
import pyproj
import requests
import shapely.geometry
import shapely.geometry
import shapely.ops
from geopandas import GeoSeries
from toolz import curry, pipe

from tile2net.logger import logger

if False:
    from tile2net.raster.tile import Tile

class SourceMeta(ABCMeta):
    catalog: dict[str, Type['Source']] = {}

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
        cls: Type[Source],
        item: list[float] | str | shapely.geometry.base.BaseGeometry,
    ) -> Optional['Source']:
        original = item
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
                .to_crs('epsg:3857')
                .loc[lambda x: x.intersects(item)]
            )
            if matches.empty:
                return None
                # raise KeyError(f'No source found for {item}')
            items = (
                matches.intersection(item)
                .area
                .__truediv__(matches.area)
                # .idxmax()
            )
            if len(items) > 1:
                logger.info(
                    f'Found multiple sources for the location, in descending IOU: '
                    f'{items.sort_values(ascending=False).index.tolist()}'
                )
            item = items.idxmax()

        if isinstance(item, str):
            if item not in cls.catalog:
                return None
                # raise KeyError(f'No source found for {item}')
            source = cls.catalog[item]

        else:
            raise TypeError(f'Invalid type {type(original)} for {original}')
        return source()

    def __init__(self: Type[Source], name, bases, attrs, **kwargs):
        # super(type(self), self).__init__(name, bases, attrs, **kwargs)
        super().__init__(name, bases, attrs)
        if (
                ABC not in bases
                and kwargs.get('init', True)
        ):
            if self.name is None:
                raise ValueError(f'{self} must have a name')
            if self.name in self.catalog:
                raise ValueError(f'{self} name already in use')
            self.catalog[self.name] = self

            # for attr in class_attr.relevant_to(self):
            #     attr.__get__(None, self)

class Source(ABC, metaclass=SourceMeta):
    name: str = None
    coverage: GeoSeries = None
    zoom: int = None
    extension = 'png'
    tiles: str = None
    tilesize: int = 256

    def __getitem__(self, item: Iterator[Tile]):
        tiles = self.tiles
        yield from (
            tiles.format(z=tile.zoom, y=tile.ytile, x=tile.xtile)
            for tile in item
        )

    def __bool__(self):
        return True

    def __repr__(self):
        return f'<{self.__class__.__qualname__} {self.name} at {hex(id(self))}>'

    def __str__(self):
        return self.name

    def __init_subclass__(cls, **kwargs):
        # complains if gets kwargs
        super().__init_subclass__()

class class_attr:
    # caches properties to the class if class is not abc
    cache = WeakKeyDictionary()

    @classmethod
    def relevant_to(cls, item: SourceMeta) -> set[class_attr]:
        res = {
            attr
            for subclass in item.mro()
            if subclass in cls.cache
            for attr in cls.cache[subclass]
        }
        return res

    def __get__(self, instance, owner: Source | SourceMeta):
        result = self.func(owner)
        type.__setattr__(owner, self.name, result)
        return result

    def __init__(self, func):
        if not isinstance(func, property):
            raise TypeError(f'{func} must be a property')
        func = func.fget
        self.func = func
        functools.update_wrapper(self, func)
        return

    def __set_name__(self, owner, name):
        self.name = name
        self.cache.setdefault(owner, set()).add(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.__name__}>'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

# noinspection PyPropertyDefinition
class ArcGis(Source, ABC):
    server: str = None

    @class_attr
    @property
    def layer_info(cls):
        res = pipe(
            requests.get(cls.metadata).text,
            json.loads,
        )
        return res

    @class_attr
    @property
    def coverage(cls):
        crs = cls.layer_info['spatialReference']['latestWkid']
        res = GeoSeries([shapely.geometry.box(
            cls.layer_info['fullExtent']['xmin'],
            cls.layer_info['fullExtent']['ymin'],
            cls.layer_info['fullExtent']['xmax'],
            cls.layer_info['fullExtent']['ymax'],
        )], crs=crs).to_crs('epsg:4326')
        return res

    @class_attr
    @property
    def zoom(cls):
        try:
            res = cls.layer_info['maxLOD']
        except KeyError:
            res = max(cls.layer_info['tileInfo']['lods'], key=lambda x: x['level'])['level']
        res = min(res, 20)
        return res

    @class_attr
    @property
    def metadata(cls):
        return cls.server + '?f=json'

    @class_attr
    @property
    def tiles(cls):
        return cls.server + '/tile/{z}/{y}/{x}'

class NewYorkCity(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_-_2020/MapServer'
    name = 'nyc'

class NewYork(ArcGis):
    server = 'https://orthos.its.ny.gov/arcgis/rest/services/wms/2020/MapServer'
    name = 'ny'

class Massachusetts(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/USGS_Orthos_2019/MapServer'
    name = 'ma'

class KingCountyWashington(ArcGis):
    server = 'https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2021/MapServer'
    name = 'king'

class WashingtonDC(ArcGis):
    server = 'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer'
    name = 'dc'
    tilesize = 512
    extension = 'jpeg'

    def __getitem__(self, item: Iterator[Tile]):
        for tile in item:
            top, left, bottom, right = tile.transformProject(tile.crs, 3857)
            yield (
                f'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer'
                f'/exportImage?f=image&bbox={bottom}%2C{right}%2C{top}%2C{left}'
                f'&imageSR=102100&bboxSR=102100&size=512%2C512'
            )

    @class_attr
    @property
    def zoom(cls):
        return 19
        # return 20

class LosAngeles(ArcGis):
    server = 'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Aerial_2014/MapServer'
    name = 'la'

class WestOregon(ArcGis, init=False):
    server = 'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2018/OSIP_2018_WM/ImageServer'
    name = 'w_or'
    extension = 'jpeg'
    # todo: ssl incorrectly configured; come back later

class EastOregon(ArcGis, init=False):
    server = 'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2017/OSIP_2017_WM/ImageServer'
    name = 'e_or'
    extension = 'jpeg'

class NewJersey(ArcGis):
    server = 'https://maps.nj.gov/arcgis/rest/services/Basemap/Orthos_Natural_2020_NJ_WM/MapServer'
    name = 'nj'

if __name__ == '__main__':
    lat1, lon1 = 40.59477460446395, -73.96014473965148
    lat2, lon2 = 40.636082070035755, -73.92478249851513
    coverage = [
        min(lat1, lat2),
        max(lat1, lat2),
        min(lon1, lon2),
        max(lon1, lon2),
    ]
    # source = Source[coverage]

if __name__ == '__main__':
    NewYorkCity.metadata
    NewYorkCity.metadata