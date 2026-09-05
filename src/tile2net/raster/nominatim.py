from __future__ import annotations
import numpy as np
from geopy.geocoders.base import DEFAULT_SENTINEL
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd
from functools import cache, cached_property, lru_cache, partial, partialmethod, update_wrapper, wraps
from collections import UserDict, UserList, UserString, defaultdict, deque, namedtuple, defaultdict, deque
from functools import cached_property, lru_cache, partial, partialmethod, reduce, singledispatch, singledispatchmethod, \
    update_wrapper, wraps
from typing import Any, Callable, Optional, Union, Type, TypeVar, Generic, Protocol, Annotated, Literal, Final, \
    ClassVar, TypeAlias, NamedTuple, TypedDict, Iterable, Iterator, Generator, cast, overload, TYPE_CHECKING, Self
from shapely import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, GeometryCollection, box
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from pandas.api.extensions import ExtensionArray

import copy

import json
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from geopy.geocoders import Nominatim as _Nominatim
from geopy.location import Location

import geopy.geocoders

CACHE_DIRECTORY_ENV = 'TILE2NET_CACHE_DIR'
SQLITE_PATH_ENV = 'TILE2NET_NOMINATIM_CACHE'


def cache_directory() -> Path:
    """User-writable cache directory, mirroring raster.weights.weights_directory."""
    configured = os.environ.get(CACHE_DIRECTORY_ENV)
    if configured:
        return Path(configured).expanduser()

    xdg_cache = os.environ.get('XDG_CACHE_HOME')
    if xdg_cache:
        cache_root = Path(xdg_cache).expanduser()
    elif sys.platform == 'darwin':
        cache_root = Path.home() / 'Library' / 'Caches'
    elif (
            os.name == 'nt'
            and os.environ.get('LOCALAPPDATA')
    ):
        cache_root = Path(os.environ['LOCALAPPDATA'])
    else:
        cache_root = Path.home() / '.cache'
    return cache_root / 'tile2net'


class Cache:
    parent: Optional[Nominatim]

    def __set_name__(self, owner, name):
        self.__name__ = name
        self._singleton: Self = self

    def __get__(
            self,
            instance: Optional[Nominatim],
            owner
    ) -> Self:
        out = copy.copy(self)
        out.parent = instance
        return out

    def __init__(
            self,
            func=None,
            read=True,
            write=False,
    ):
        self.read = read
        self.write = write

    def _write(self):
        ...

    def _read(self):
        ...

    def geocode(self):
        ...

    def reverse(self):
        ...


class JSON(Cache):
    _singleton: Self

    @property
    def path(self):
        configured = os.environ.get(SQLITE_PATH_ENV)
        if self._singleton is not self:
            return self._singleton.path
        path = Path(__file__, '../', 'nominatim.json').resolve()
        self._path = path
        return path

    def __init__(
            self,
            func=None,
            read=True,
            write=False,
    ):
        super().__init__(func, read, write)

    def _write(self):
        ...

    def _read(self):
        ...

    def geocode(self):
        ...

    def reverse(self):
        ...


class SQLite:
    _singleton: Self

    @property
    def path(self):
        configured = os.environ.get(SQLITE_PATH_ENV)
        if self._singleton is not self:
            return self._singleton.path
        if configured:
            path = Path(configured).expanduser()
        else:
            path = cache_directory() / 'nominatim.sqlite'
        self._path = path
        return path


    def __init__(
            self,
            func=None,
            read=True,
            write=True,
    ):
        super().__init__(func, read, write)

    def _write(self):
        ...

    def _read(self):
        ...

    def geocode(self):
        ...

    def reverse(self):
        ...


class Nominatim(geopy.geocoders.Nominatim):
    @JSON
    def json(self):
        return Cache().__get__(self, type(self))

    @SQLite
    def sqlite(self):
        return Cache().__get__(self, type(self))

    def geocode(
            self,
            query,
            *,
            exactly_one=True,
            timeout=DEFAULT_SENTINEL,
            limit=None,
            addressdetails=False,
            language=False,
            geometry=None,
            extratags=False,
            country_codes=None,
            viewbox=None,
            bounded=False,
            featuretype=None,
            namedetails=False
    ):
        ...

    def reverse(
            self,
            query,
            *,
            exactly_one=True,
            timeout=DEFAULT_SENTINEL,
            language=False,
            addressdetails=True,
            zoom=None,
            namedetails=False,
    ):
        ...



if __name__ == '__main__':
    ...
