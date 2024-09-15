from __future__ import annotations

import functools
import json
import pathlib
import warnings
from abc import ABC, ABCMeta
from functools import cached_property, wraps
from typing import Iterator, Optional, Type
from typing import TypeVar
from weakref import WeakKeyDictionary

import geopandas as gpd
import pandas as pd
import requests
import shapely.geometry
import shapely.geometry
import shapely.ops
from geopandas import GeoDataFrame
from geopandas import GeoSeries

from tile2net.logger import logger
from tile2net.raster.geocode import GeoCode

if False:
    from tile2net.raster.tile import Tile


class Coverage:
    @cached_property
    def file(self) -> pathlib.Path:
        return pathlib.Path(
            __file__, '..', '..', 'resources', 'coverage.feather'
        ).resolve()

    def __get__(self, instance, owner: SourceMeta) -> GeoSeries:
        if self.file.exists():
            coverage = gpd.read_feather(self.file)
            # if not coverage.index.symmetric_difference(owner.catalog.keys()):
            if (
                    coverage.index
                            .symmetric_difference(owner.catalog.keys())
                            .empty
            ):
                # noinspection PyTypeChecker
                return coverage.geometry
        coverages: list[GeoSeries] = []
        for source in owner.catalog.values():
            try:
                axis = pd.Index([source.name] * len(source.coverage), name='source')
                coverage = (
                    source.coverage
                    .set_crs('epsg:4326')
                    .set_axis(axis)
                )
            except Exception as e:
                logger.error(
                    f'Could not get coverage for {source.name}, skipping:\n'
                    f'{e}'
                )
            else:
                coverages.append(coverage)

        self.file.parent.mkdir(parents=True, exist_ok=True)
        coverage = GeoDataFrame({
            'geometry': pd.concat(coverages),
        })
        coverage.to_feather(self.file)
        coverage = coverage.geometry
        setattr(owner, self.__name__, coverage)
        return coverage

    def __set_name__(self, owner, name):
        self.__name__ = name


class SourceNotFound(Exception):
    ...


T = TypeVar('T')


def not_found_none(func: T) -> T:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SourceNotFound as e:
            return None

    return wrapper


class SourceMeta(ABCMeta):
    catalog: dict[str, Type[Source]] = {}
    coverage = Coverage()

    @not_found_none
    def __getitem__(
            cls: Type[Source],
            item: list[float] | str | shapely.geometry.base.BaseGeometry,
    ) -> Optional['Source']:
        # todo: index index for which sources contain keyword
        if item in cls.catalog:
            return cls.catalog[item]()
        # select where geometry intersects the coverage

        matches: GeoSeries = SourceMeta.coverage.geometry
        geocode = GeoCode.from_inferred(item)
        loc = matches.intersects(geocode.polygon)
        if (
            not loc.any()
            and 'address' in geocode.__dict__
        ):
            # user must've been lazy; compute a new polygon
            del geocode.address
            _ = geocode.address
            del geocode.nwse
            del geocode.wsen
            del geocode.polygon
            loc = matches.intersects(geocode.polygon)
            if not loc.any():
                raise SourceNotFound
        matches = matches.loc[loc]

        # to resolve discrepancies, select where keyword is in the address
        loc = []
        for name in matches.index:
            keyword: str | tuple[str]
            keyword = cls.catalog[name].keyword
            if isinstance(keyword, str):
                loc.append(keyword.casefold() in geocode.address.casefold())
            else:
                append = any(
                    word.casefold() in geocode.address.casefold()
                    for word in keyword
                )
                loc.append(append)

        if (
            not any(loc)
            and 'address' in geocode.__dict__
        ):
            # user must've been lazy; compute a new address
            loc = []
            del geocode.address
            _ = geocode.address
            for name in matches.index:
                keyword: str | tuple[str]
                keyword = cls.catalog[name].keyword
                if isinstance(keyword, str):
                    loc.append(keyword.casefold() in geocode.address.casefold())
                else:
                    append = any(
                        word.casefold() in geocode.address.casefold()
                        for word in keyword
                    )
                    loc.append(append)

        if any(loc):
            matches = matches.loc[loc]
        elif 'address' not in geocode.__dict__:
            raise SourceNotFound
        else:
            logger.warning(
                f'No keyword matches found for {item=} using '
                f'{geocode.address=}; the result may be inaccurate',
            )
        if len(matches) == 1:
            return cls.catalog[matches.index[0]]()

        # bboxs = matches.intersection(geocode.polygon).area
        # bboxs /= matches.area

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bboxs = matches.intersection(geocode.polygon).area / matches.area
        item = bboxs.idxmax()
        if len(bboxs) > 1:
            logger.info(
                f'Found multiple sources for the location, in descending IOU: '
                f'{bboxs.sort_values(ascending=False).index.tolist()} and '
                f'chose {item} ({cls.catalog[item].keyword})'
            )
        if isinstance(item, str):
            if item not in cls.catalog:
                raise SourceNotFound
            source = cls.catalog[item]
        else:
            raise TypeError(f'Invalid type {type(item)} for {item}')
        return source()

    # def __init__(self: Type[Source], name, bases, attrs, **kwargs):
    #     super().__init__(name, bases, attrs)
    #     if not getattr(self, 'ignore', False):
    #         if self.name is None:
    #             raise ValueError(f'{self} must have a name')
    #         if self.name in self.catalog:
    #             raise ValueError(f'{self} name already in use')
    #         self.catalog[self.name] = self


class Source(ABC, metaclass=SourceMeta):
    name: str = None  # name of the source
    coverage: GeoSeries = None  # coverage that contains a polygon representing the coverage
    zoom: int = None  # xyz tile zoom level
    extension = 'png'
    tiles: str = None
    tilesize: int = 256  # pixels per tile side
    keyword: str  # required match when reverse geolocating address from point
    dropword: str = None  # if result contains this word, it is not a match

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
        super().__init_subclass__()
        if (
            not getattr(cls, 'ignore', False)
            and ABC not in cls.__bases__
        ):
            if cls.name is None:
                raise ValueError(f'{cls} must have a name')
            if cls.name in cls.catalog:
                raise ValueError(f'{cls} name already in use')
            cls.catalog[cls.name] = cls

    def __eq__(self, other):
        if (
                isinstance(other, Source)
                or isinstance(other, SourceMeta)
        ):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented


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
        response = requests.get(cls.metadata)
        response.raise_for_status()
        text = response.text
        try:
            res = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f'Could not parse JSON from stdin: {e}')
            logger.error(f'{cls.metadata=}; {cls.server=}')
            logger.error(f'JSON: {text}')
            raise
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


"""
Note: sometimes we get something like Spring Hill, Maury County,
 Middle Tennessee, so it's important to have multiple keywords
 if it's not just a state or major city
"""


class NewYorkCity(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_-_2020/MapServer'
    name = 'nyc'
    keyword = 'New York City', 'City of New York'


class NewYork(ArcGis):
    server = 'https://orthos.its.ny.gov/arcgis/rest/services/wms/2020/MapServer'
    name = 'ny'
    keyword = 'New York'


class Massachusetts(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/USGS_Orthos_2019/MapServer'
    name = 'ma'
    keyword = 'Massachusetts'
    extension = 'jpg'


class KingCountyWashington(ArcGis):
    server = 'https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2021/MapServer'
    name = 'king'
    keyword = 'King County, Washington', 'King County'


class WashingtonDC(ArcGis):
    # ignore = True
    server = 'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer'
    name = 'dc'
    tilesize = 512
    extension = 'jpeg'
    keyword = 'District of Columbia', 'DC'

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


class LosAngeles(ArcGis):
    server = 'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Aerial_2014/MapServer'
    name = 'la'
    keyword = 'Los Angeles'

    # to test case where a source raises an error due to metadata failure
    #   other sources should still function
    # @class_attr
    # @property
    # def metadata(cls):
    #     raise NotImplementedError


# class WestOregon(ArcGis):
#     ignore = True
#     server = 'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2018/OSIP_2018_WM/ImageServer'
#     name = 'w_or'
#     extension = 'jpeg'
#     keyword = 'Oregon'
#     # todo: ssl incorrectly configured; come back later
#

# class EastOregon(ArcGis, init=False):
#     ignore = True
#
#     server = 'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2017/OSIP_2017_WM/ImageServer'
#     name = 'e_or'
#     extension = 'jpeg'
#     keyword = 'Oregon'

# class Oregon(ArcGis):
#     server = 'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2022/OSIP_2022_WM/ImageServer'
#     name = 'or'
#     extension = 'jpeg'
#     keyword = 'Oregon'
#

class NewJersey(ArcGis):
    server = 'https://maps.nj.gov/arcgis/rest/services/Basemap/Orthos_Natural_2020_NJ_WM/MapServer'
    name = 'nj'
    keyword = 'New Jersey'


class SpringHillTN(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/tF0XsRR9ptiKNVW2/arcgis/rest/services/Spring_Hill_Imagery_WGS84/MapServer'
    name = 'sh_tn'
    keyword = 'Spring Hill, Tennessee', 'Spring Hill'


class Virginia(ArcGis):
    """Data from https://vgin.vdem.virginia.gov/pages/orthoimagery"""
    server = "https://gismaps.vdem.virginia.gov/arcgis/rest/services/VBMP_Imagery/MostRecentImagery_WGS/MapServer/"
    name = "va"
    keyword = "Virginia"
    box = shapely.geometry.box(-83.6753, 36.5407, -75.1664, 39.4660)
    coverage = GeoSeries(box, crs='epsg:4326')


if __name__ == '__main__':
    assert Source['New Brunswick, New Jersey'] == NewJersey
    assert Source['New York City'] == NewYorkCity
    assert Source['New York'] in (NewYorkCity, NewYork)
    assert Source['Massachusetts'] == Massachusetts
    assert Source['King County, Washington'] == KingCountyWashington
    assert Source['Washington, DC'] == WashingtonDC
    assert Source['Los Angeles'] == LosAngeles
    assert Source['Jersey City'] == NewJersey
    assert Source['Hoboken'] == NewJersey
    assert Source["Spring Hill, TN"] == SpringHillTN
    assert Source['Oregon'] == Oregon
    assert Source['Virginia'] == Virginia
