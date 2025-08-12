from __future__ import annotations

import functools
import json
import pathlib
import warnings
from abc import ABC
from functools import cached_property
from typing import *
from typing import Optional
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
from shapely import box
from shapely.geometry import box

from tile2net.grid.cfg.logger import logger
from tile2net.raster.geocode import GeoCode

if False:
    from .ingrid import InGrid


class SourceNotFound(Exception):
    ...


T = TypeVar('T')


class Coverage:
    @cached_property
    def file(self) -> pathlib.Path:
        return pathlib.Path(
            __file__, '..', '..', 'resources', 'coverage.feather'
        ).resolve()

    def __get__(
            self,
            instance,
            owner: type[Source]
    ):
        if self.file.exists():
            coverage = gpd.read_feather(self.file)
            # if not coverage.index.symmetric_difference(owner.catalog.keys()):
            empty = (
                coverage.index
                .symmetric_difference(owner.catalog.keys())
                .empty
            )
            if empty:
                # noinspection PyTypeChecker
                return coverage.geometry
        coverages: list[GeoSeries] = []
        for source in owner.catalog.values():
            # source is WashingtonDC

            if source.outdated:
                continue
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

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...



def __init__(
        self: cls_attr,
        func: Callable[..., T],
):
    # if not isinstance(func, property):
    #     raise TypeError(f'{func} must be a property')
    # func = func.fget
    if isinstance(func, property):
        func = func.fget
    self.func = func
    functools.update_wrapper(self, func)


class cls_attr(
    # Generic[T],
):
    __wrapped__: Callable = None

    def _get(
            self: cls_attr,
            instance,
            owner: Source | type[Source]
    ) -> T:
        result = self.__wrapped__(owner)
        type.__setattr__(owner, self.name, result)
        return result

    cache = WeakKeyDictionary()
    locals().update(
        __get__=_get,
        __init__=__init__,
    )

    if False:
        def __new__(cls, func: Callable[..., T]) -> T:
            ...

    @classmethod
    def relevant_to(cls, item: type[Source]) -> set[cls_attr]:
        res = {
            attr
            for subclass in item.mro()
            if subclass in cls.cache
            for attr in cls.cache[subclass]
        }
        return res

    def __set_name__(self, owner, name):
        self.name = name
        self.cache.setdefault(owner, set()).add(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.__name__}>'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


# class SourceABC(
#     # ...
#     type,
# ):
#     def __eq__(self, other):
#         return True
#





# noinspection PyMethodParameters
class Source(
    ABC,
    # metaclass=SourceABC,
):
    grid: InGrid
    catalog: dict[str, type[Source]] = {}

    outdated: bool = False

    def _get(
            self: Source,
            instance: InGrid,
            owner: type[InGrid],
    ) -> Source:
        """Return the source object for the grid instance."""
        try:
            result = instance.__dict__[self.__name__]
            result.grid = instance
            result.InGrid = owner
        except KeyError as e:
            msg = (
                f'Source has not yet been set. To set the source, you '
                f'must call `InGrid.with_source()`.'
            )
            raise KeyError(msg) from e
        return result

    locals().update(
        __get__=_get,
    )

    """
    Catalog of all sources available in the package.
    This is used to look up sources by name.
    """

    @cls_attr
    def name(cls) -> str:
        """Short name of the source, e.g. `nyc` for New York City."""

    @Coverage
    def coverage(cls) -> GeoSeries:
        ...

    # @cls_attr
    # def coverage(cls) -> Union[
    #     GeoSeries,
    #     GeoDataFrame,
    # ]:
    #     """
    #     Spatial coverage of the source, to be used for deciding which
    #     sources are relevant to an area.
    #     """

    @cls_attr
    def zoom(cls) -> int:
        """
        Default XYZ zoom level for the source.
        Our model performs best with a zoom of at least 19.
        """

    @cls_attr
    def extension(cls) -> str:
        """File extension for the source, e.g. 'png' or 'jpg'."""
        return 'png'

    @cls_attr
    def dimension(cls):
        """Default dimension of the source grid, e.g. 256 pixels."""
        return 256

    @cls_attr
    def keyword(cls) -> Union[
        str,
        tuple[str, ...],
    ]:
        """
        A keyword is a required match in the reverse geocode to resolve
        discrepancies.
        """

    @cls_attr
    def dropword(cls) -> Union[
        str,
        tuple[str, ...]
    ]:
        """
        A dropword is the reverse of a keyword. If a reverse geocode
        contains this, the source is not relevant.
        """

    @cls_attr
    def year(cls) -> int:
        """Year of the data"""

    @cls_attr
    def template(cls) -> str:
        """Template for formatting the URL of the grid."""

    def __set__(
            self,
            instance: InGrid,
            value,
    ):
        """Set the source object for the grid instance."""
        instance.__dict__[self.__name__] = value

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __delete__(
            self,
            instance: InGrid,
    ):
        """Delete the source object for the grid instance."""
        if hasattr(instance, '_source'):
            del instance._source

    def __repr__(self):
        return f'<{self.__class__.__qualname__} {self.name} at {hex(id(self))}>'

    def __init__(
            self,
            *args,
            **kwargs
    ):
        ...

    @property
    def urls(self) -> pd.Series:
        """Given some grid, return the URL for the images"""
        grid = self.grid
        temp = self.template
        zoom = grid.zoom
        it = zip(grid.ytile, grid.xtile)
        data = [
            temp.format(z=zoom, y=ytile, x=xtile)
            for ytile, xtile in it
        ]
        result = pd.Series(data, index=grid.index, name='url')
        return result

    @classmethod
    def from_name(
            cls,
            name: str,
    ) -> Optional['Source']:
        """
        Return a source by its name.
        If the source is not found, return None.
        """
        if name in cls.catalog:
            return cls.catalog[name]()
        raise SourceNotFound(f'Source {name} not found.')

    @classmethod
    # @not_found_none
    def from_inferred(
            cls,
            # item: list[float] | str | shapely.geometry.base.BaseGeometry,
            item: Union[
                list[float],
                str,
                shapely.geometry.base.BaseGeometry,
                gpd.GeoSeries,
                gpd.GeoDataFrame,
            ]
    ) -> Optional['Source']:
        # todo: index index for which sources contain keyword

        matches: GeoSeries = Source.coverage.geometry
        if isinstance(item, (GeoSeries, GeoDataFrame)):
            infer = item.geometry.iat[0].centroid
        else:
            infer = item
        geocode = GeoCode.from_inferred(infer)
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        if (
                not getattr(cls, 'ignore', False)
                and ABC not in cls.__bases__
        ):
            if cls.name is None:
                raise ValueError(f'{cls} must have a name')
            if cls.name in cls.catalog:
                raise ValueError(f'{cls} name {cls.name} already in use')
            cls.catalog[cls.name] = cls

    def __eq__(self, other):
        if (
            isinstance(other, Source)
        ) or (
            isinstance(other, type)
            and issubclass(other, Source)
        ):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented


# noinspection PyMethodParameters
class ArcGis(
    Source,
    ABC
):
    @cls_attr
    def server(cls) -> str:
        """ base URL for the ArcGIS server."""

    @cls_attr
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

    @cls_attr
    def coverage(cls):
        crs = cls.layer_info['spatialReference']['latestWkid']
        res = GeoSeries([shapely.geometry.box(
            cls.layer_info['fullExtent']['xmin'],
            cls.layer_info['fullExtent']['ymin'],
            cls.layer_info['fullExtent']['xmax'],
            cls.layer_info['fullExtent']['ymax'],
        )], crs=crs).to_crs('epsg:4326')
        return res

    @cls_attr
    def zoom(cls):
        try:
            res = cls.layer_info['maxLOD']
        except KeyError:
            res = max(cls.layer_info['tileInfo']['lods'], key=lambda x: x['level'])['level']
        res = min(res, 20)
        return res

    @cls_attr
    def metadata(cls):
        return cls.server + '?f=json'

    @cls_attr
    def template(cls):
        return cls.server + '/tile/{z}/{y}/{x}'


class WashingtonDC(ArcGis):
    # ignore = True
    server = 'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer'
    name = 'dc'
    gridize = 512
    extension = 'jpeg'
    keyword = 'District of Columbia', 'DC'
    year = 2021

    template = (
        'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer'
        '/exportImage?f=image&bbox={bottom}%2C{right}%2C{top}%2C{left}'
        '&imageSR=102100&bboxSR=102100&size=512%2C512'
    )

    def __getitem__(self, item: InGrid) -> pd.Series[str]:
        bounds = item.bounds
        it = zip(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy)
        template = self.template
        data = [
            template.format(
                bottom=miny,
                right=maxx,
                top=maxy,
                left=minx
            )
            for minx, miny, maxx, maxy in it
        ]
        result = pd.Series(data, index=item.index, name='url')
        return result


class NewYorkCity(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_-_2020/MapServer'
    name = 'nyc'
    keyword = 'New York City', 'City of New York'
    year = 2020


class NewYork(ArcGis):
    server = 'https://orthos.its.ny.gov/arcgis/rest/services/wms/2020/MapServer'
    name = 'ny'
    keyword = 'New York'
    year = 2020


class Massachusetts(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/USGS_Orthos_2019/MapServer'
    name = 'ma'
    keyword = 'Massachusetts'
    extension = 'jpg'
    year = 2019


class KingCountyWashington(ArcGis):
    server = 'https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2021/MapServer'
    name = 'king'
    keyword = 'King County, Washington', 'King County'
    year = 2021


class LosAngeles(ArcGis):
    server = 'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Aerial_2014/MapServer'
    name = 'la'
    keyword = 'Los Angeles'
    year = 2014

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

# todo: Oregon also has some SSL issues
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
    year = 2020


class SpringHillTN(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/tF0XsRR9ptiKNVW2/arcgis/rest/services/Spring_Hill_Imagery_WGS84/MapServer'
    name = 'sh_tn'
    keyword = 'Spring Hill, Tennessee', 'Spring Hill'
    year = 2020


class Virginia(ArcGis):
    """Data from https://vgin.vdem.virginia.gov/pages/orthoimagery"""
    server = "https://gismaps.vdem.virginia.gov/arcgis/rest/services/VBMP_Imagery/MostRecentImagery_WGS/MapServer/"
    name = "va"
    keyword = "Virginia"
    box = shapely.geometry.box(-83.6753, 36.5407, -75.1664, 39.4660)
    coverage = GeoSeries(box, crs='epsg:4326')


class AlamedaCounty(
    Source,
):
    # https://www.arcgis.com/home/item.html?id=46db377005dc4e76bbc234c680771573
    # ignore = True
    name = 'al'
    extension = 'png'
    keyword = 'Alameda County', 'California'
    year = 2023
    server = (
        'https://svc.pictometry.com/Image/'
        '6D9E15C5-C6B4-4ACB-A244-4C44ECA33D90/'
        'wmts/PICT-CAALAM20-OQ74EOAkBw/default/GoogleMapsCompatible'
    )
    zoom = 20

    @cls_attr
    @property
    def metadata(cls):
        return 'https://svc.pictometry.com/Image/6D9E15C5-C6B4-4ACB-A244-4C44ECA33D90/wmts?SERVICE=WMTS&REQUEST=GetCapabilities&VERSION=1.0.0'

    @cls_attr
    @property
    def grid(cls):
        return cls.server + '/{z}/{x}/{y}.png'

    @cls_attr
    @property
    def coverage(cls):
        res = GeoSeries([
            box(
                -122.345118999,  # xmin (W)
                37.451422681,  # ymin (S)
                -121.462793698,  # xmax (E)
                37.912241999  # ymax (N)
            )
        ], crs='EPSG:4326')
        return res


class SanFranciscoBase(
    Source,
    ABC
):
    # static attributes
    outdated: bool = True
    extension: str = "png"
    keyword: tuple[str, str] = ("San Francisco", "California")
    zoom: int = 20
    dimension: int = 512  # tile size in px
    server: str

    # ── lazily-computed class attributes (cached via cls_attr) ────────────────
    @cls_attr
    def metadata(cls) -> str:  # noqa: D401  (no docstrings requested)
        return f"{cls.server}/tiles.json"

    @cls_attr
    def template(cls) -> str:
        return f"{cls.server}/{{z}}/{{x}}/{{y}}.png"

    @cls_attr
    def coverage(cls) -> GeoSeries:
        return (
            GeoSeries(
                [
                    box(
                        -122.514926,  # xmin (W)
                        37.708075,  # ymin (S)
                        -122.356779,  # xmax (E)
                        37.832371  # ymax (N)
                    )
                ],
                crs="EPSG:4326",
            )
        )


class SanFrancisco2014(SanFranciscoBase):
    name: str = "sf2014"
    year: int = 2014
    server: str = "https://tile.sf.gov/api/tiles/p2014_rgb8cm"


class SanFrancisco2017(SanFranciscoBase):
    name: str = "sf2017"
    year: int = 2017
    server: str = "https://tile.sf.gov/api/tiles/p2017_rgb8cm"


class SanFrancisco2018(SanFranciscoBase):
    name: str = "sf2018"
    year: int = 2018
    server: str = "https://tile.sf.gov/api/tiles/p2018_rgb8cm"


class SanFrancisco2019(SanFranciscoBase):
    name: str = "sf2019"
    year: int = 2019
    server: str = "https://tile.sf.gov/api/tiles/p2019_rgb8cm"


class SanFrancisco2020(SanFranciscoBase):
    name: str = "sf2020"
    year: int = 2020
    server: str = "https://tile.sf.gov/api/tiles/p2020_rgb8cm"


class SanFrancisco2023(SanFranciscoBase):
    name: str = "sf2023"
    year: int = 2023
    server: str = "https://tile.sf.gov/api/tiles/p2023_rgb8cm"


class SanFrancisco2024(SanFranciscoBase):
    name: str = "sf2024"
    year: int = 2024
    server: str = "https://tile.sf.gov/api/tiles/p2024_rgb8cm"
    outdated: bool = False  # mark most-recent layer as current


class MaineOrthoBase(
    ArcGis,
    ABC
):
    """
    Shared config for Maine GeoLibrary statewide imagery.
    (Everything else—coverage, zoom, template—comes from ArcGis.)
    """
    keyword: str = "Maine"
    extension: str = "jpeg"  # ArcGIS ImageServer delivers JPEG by default


class Maine2021(MaineOrthoBase):
    server: str = (
        "https://gis.maine.gov/arcgis/rest/services/"
        "imageryBaseMapsEarthCover/orthoRegional2021/ImageServer"
    )
    name: str = "me2021"
    year: int = 2021
    outdated: bool = True  # superseded by 2022 imagery


class Maine2022(MaineOrthoBase):
    server: str = (
        "https://gis.maine.gov/arcgis/rest/services/"
        "imageryBaseMapsEarthCover/orthoRegional2022/ImageServer"
    )
    name: str = "me2022"
    year: int = 2022
    outdated: bool = False  # current statewide layer
    gridize: int = 512
    extension: str = "jpeg"
    zoom=19

    # ArcGIS ExportImage template – bbox in Web-Mercator metres
    template: str = (
        "https://gis.maine.gov/arcgis/rest/services/"
        "imageryBaseMapsEarthCover/orthoRegional2022/ImageServer"
        "/exportImage?f=image"
        "&bbox={left}%2C{bottom}%2C{right}%2C{top}"
        "&bboxSR=102100&imageSR=102100"
        "&size=256%2C256"
    )

    @property
    def urls(self) -> pd.Series:
        # project tile geometries to EPSG 102100, then build URLs
        item = self.grid
        bounds = item.to_crs(102100).bounds
        tmpl = self.template

        urls = [
            tmpl.format(
                left=minx, bottom=miny,
                right=maxx, top=maxy,
            )
            for minx, miny, maxx, maxy in zip(
                bounds.minx, bounds.miny, bounds.maxx, bounds.maxy
            )
        ]
        result = pd.Series(urls, index=item.index, name="url")
        return result


if __name__ == '__main__':
    assert Source.from_inferred('New Brunswick, New Jersey') == NewJersey
    assert Source.from_inferred('New York City') == NewYorkCity
    assert Source.from_inferred('New York') in (NewYorkCity, NewYork)
    assert Source.from_inferred('Massachusetts') == Massachusetts
    assert Source.from_inferred('King County, Washington') == KingCountyWashington
    assert Source.from_inferred('Washington, DC') == WashingtonDC
    assert Source.from_inferred('Los Angeles') == LosAngeles
    assert Source.from_inferred('Jersey City') == NewJersey
    assert Source.from_inferred('Hoboken') == NewJersey
    assert Source.from_inferred("Spring Hill, TN") == SpringHillTN
    assert Source.from_inferred('Virginia') == Virginia
    assert Source.from_inferred('Berkeley, California') == AlamedaCounty
    assert Source.from_inferred('Fremont, California') == AlamedaCounty
    assert Source.from_inferred('Oakland, California') == AlamedaCounty
    assert Source.from_inferred('San Francisco, California') == SanFrancisco2024
    assert Source.from_inferred('Bangor, Maine') == Maine2022

    Source.catalog
    Source.coverage.geometry

