from __future__ import annotations

import functools
import json
import pathlib
import warnings
import xml.etree.ElementTree as ET
from abc import ABC
from functools import cached_property
from typing import *
from typing import Iterator, Optional, Iterable, TypeVar
from urllib.parse import urlencode
from weakref import WeakKeyDictionary

import geopandas as gpd
import pandas as pd
import requests
import shapely.geometry
import shapely.geometry
import shapely.ops
from geopandas import GeoDataFrame, GeoSeries
from requests.adapters import HTTPAdapter
from shapely import box, wkt
from urllib3.util.retry import Retry

from tile2net.logger import logger
from tile2net.raster.geocode import GeoCode

if False:
    from .ingrid import InGrid


class SourceNotFound(Exception):
    ...


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
                coverage = source.coverage
                axis = pd.Index([source.name] * len(coverage), name='source')
                coverage = (
                    coverage
                    .set_crs('epsg:4326')
                    .set_axis(axis)
                )
            except Exception as e:
                logger.error(
                    f'Could not get coverage for {source.name},'
                    f' skipping:\n\t'
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


T = TypeVar('T')


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
            item: Union[
                list[float],
                str,
                shapely.geometry.base.BaseGeometry,
                gpd.GeoSeries,
                gpd.GeoDataFrame,
            ]
    ) -> Optional['Source']:
        if isinstance(item, str):
            try:
                return cls.from_name(item)
            except SourceNotFound:
                ...

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
    server = 'https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer'
    name = 'nyc'
    keyword = 'New York City', 'City of New York'
    year = 2024
    coverage = wkt.loads("POLYGON ((-73.69048 40.4889, -73.69048 40.92834, -74.27615 40.92834, -74.27615 40.4889, -73.69048 40.4889))")
    coverage = GeoSeries(coverage, crs='epsg:4326')


class NewYork(ArcGis):
    server = 'https://orthos.its.ny.gov/arcgis/rest/services/wms/2024/MapServer'
    name = 'ny'
    keyword = 'New York'
    year = 2024
    coverage = wkt.loads("POLYGON ((-73.25509 40.48341, -73.25509 45.03931, -79.77922 45.03931, -79.77922 40.48341, -73.25509 40.48341))")
    coverage = GeoSeries(coverage, crs='epsg:4326')


class Massachusetts(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/orthos2021/MapServer'
    name = 'ma'
    keyword = 'Massachusetts'
    extension = 'jpg'
    year = 2021
    coverage = wkt.loads("POLYGON ((-69.92466 41.2265, -69.92466 42.89199, -73.51979 42.89199, -73.51979 41.2265, -69.92466 41.2265))")
    coverage = GeoSeries(coverage, crs='epsg:4326')


class KingCountyWashington(ArcGis):
    server = 'https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2023/MapServer'
    name = 'king'
    keyword = 'King County, Washington', 'King County'
    year = 2023
    coverage = wkt.loads("POLYGON ((-121.03559 47.04875, -121.03559 47.96618, -122.56958 47.96618, -122.56958 47.04875, -121.03559 47.04875))")
    coverage = GeoSeries(coverage, crs='epsg:4326')



class LosAngeles(ArcGis):
    server = 'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Aerial_2014/MapServer'
    name = 'la'
    keyword = 'Los Angeles'
    year = 2014
    coverage = wkt.loads("POLYGON ((-117.63102 33.28853, -117.63102 34.83012, -118.95937 34.83012, -118.95937 33.28853, -117.63102 33.28853))")
    coverage = GeoSeries(coverage, crs='epsg:4326')

    # to test case where a source raises an error due to metadata failure
    #   other sources should still function
    # @cls_attr
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
    coverage = wkt.loads("POLYGON ((-73.85543 38.824, -73.85543 41.3866, -75.59981 41.3866, -75.59981 38.824, -73.85543 38.824))")
    coverage = GeoSeries(coverage, crs='epsg:4326')


class SpringHillTN(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/tF0XsRR9ptiKNVW2/arcgis/rest/services/Spring_Hill_Imagery_WGS84/MapServer'
    name = 'sh_tn'
    keyword = 'Spring Hill, Tennessee', 'Spring Hill'
    year = 2020
    coverage = wkt.loads("POLYGON ((-86.75604 35.58884, -86.75604 35.85806, -87.13095 35.85806, -87.13095 35.58884, -86.75604 35.58884))")
    coverage = GeoSeries(coverage, crs='epsg:4326')


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
    outdated = False


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

class VexCel(Source, ABC):
    flip_y = False
    prefer_layers: Iterable[str] = ('wide-area', 'urban', 'urban-r', 'graysky')
    server: str = 'https://api.vexcelgroup.com/v2/ortho'
    base: str = 'https://api.vexcelgroup.com/v2/ortho/wmts'
    api_key: str = None
    timeout: int = 10
    extension = 'png'

    @cls_attr
    def _session(self) -> requests.Session:
        """
        Create a requests session with
        retries and a custom User-Agent.
        """
        # ascii-only UA to avoid header encoding issues
        s = requests.Session()
        s.headers.update({'User-Agent': 'tile2net'})
        retry = Retry(
            total=5,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset("GET", ),
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
        """ Raise an HTTPError with details from the response."""
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

    @cls_attr
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

    @cls_attr
    def layer(self):
        """ Layer chosen from the WMTS layers """
        avail = {
            layer
            for layer in self.layers
            if not layer
            .lower()
            # exclude greyscale imagery
            .endswith('-g')
        }
        for cand in self.prefer_layers:
            if cand in avail:
                return cand
        if not avail:
            raise RuntimeError('No RGB WMTS layers are available for this API key.')
        result = sorted(avail)[0]
        return result

    @cls_attr
    def template(self) -> str:
        """Encoding the WMTS URL template for the tiles"""
        query = {
            'layer': self.layer,
            'image-format': self.extension,
            'api_key': self.api_key,
        }
        q = urlencode(query)
        q += '&tile-x={x}&tile-y={y}&zoom={z}'
        result = f'{self.server}/tile?{q}'
        return result


class Maine(VexCel):
    # https://www.maine.gov/geolib/data/index.html
    name = 'maine'
    api_key = "vfa_JkRqdw7HHOis.37zZe0OHVVqPlIY91FsT9arC9uGrDalMqwMW1AX4OgcfsiwtQoxDLed9OEIxKy3Ys2lMCam9C2swLrUwNqX2KrlegBRev8MRDpkqHkbSEn0fP1aEqvoDBdePjAOO9h91.4256792737"
    keyword = 'Maine'
    coverage = wkt.loads(
        "MULTIPOLYGON (((-70.64573401557249 43.09008331966716, "
        "-70.75102474636725 43.08003225358636, "
        "-70.79761105007827 43.21973948828747, "
        "-70.98176001655037 43.36789581966831, "
        "-70.94416541205806 43.46633942318429, "
        "-71.08481999999998 45.30523999999996, "
        "-70.6600225491012 45.460222886733995, "
        "-70.30495378282376 45.914794623389334, "
        "-70.00014034695016 46.69317088478567, "
        "-69.23708614772835 47.44777598732789, "
        "-68.90478084987546 47.18479462339437, "
        "-68.23430497910455 47.3546292181218, "
        "-67.7903527492851 47.066248887717, "
        "-67.79141211614706 45.702585354182794, "
        "-67.13734351262877 45.13745189063888, "
        "-66.96465999999998 44.809699999999935, "
        "-68.03251999999992 44.32520000000004, "
        "-69.05999999999996 43.980000000000096, "
        "-70.11617000000001 43.68405, "
        "-70.64573401557249 43.09008331966716)))"
    )
    coverage = GeoSeries(coverage, crs='epsg:4326')


# todo: refactor this because SourceMeta is not used anymore

# def print_coverages(
#         simplify: Optional[float] = 5,
# ) -> None:
#     """
#     utility function to print all coverages to be hard-coded
#     get the canonical coverage series; ensure EPSG:4326 for consistency
#     """
#     coverages = SourceMeta.coverage
#     # coerce to integer decimals if provided
#     decimals: Optional[int] = None
#     if simplify is not None:
#         decimals = int(simplify)
#
#     for name, geom in coverages.items():
#         # use text-level rounding; does not alter topology/coords
#         if decimals is not None:
#             txt = geom.simplify(simplify, preserve_topology=True).wkt
#         else:
#             txt = geom.wkt
#
#         # print the source name (index) and its WKT
#         print(name)
#         print(txt)



if __name__ == '__main__':
    assert Source.from_inferred('Portland, Maine') == Maine
    assert Source.from_inferred('Maine') == Maine
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


