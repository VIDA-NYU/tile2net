from __future__ import annotations
import contextlib

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
from tile2net.grid.geocode import GeoCode

if False:
    from ..ingrid import InGrid


class Source(

):
    @property
    def files(self):
        raise NotImplementedError

    def _get(
            self: Source,
            instance: InGrid,
            owner: type[InGrid],
    ) -> Source:
        """Return the remote object for the grid instance."""
        try:
            result = instance.__dict__[self.__name__]
            result.grid = instance
            result.InGrid = owner
        except KeyError as e:
            msg = (
                f'Source has not yet been set. To set the remote, you '
                f'must call `InGrid.with_remote()`.'
            )
            raise ValueError(msg) from e
        return result

    locals().update(__get__=_get)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(
            self,
            instance: InGrid,
            value: Union[Source, str],
    ):
        if isinstance(value, Source):
            ...
        elif isinstance(value, str):
            value = self.from_str(value)
        else:
            msg = (
                f'Cannot set source with value of type {type(value)}. '
                f'Expected either a Source or a str.'
            )
            raise TypeError(msg)
        instance.__dict__[self.__name__] = value

    def __delete__(self, instance: InGrid):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            ...

    @classmethod
    def from_str(cls, value: str) -> Self:
        from .remote import Remote
        from .local import Local
        if Local.accepts(value):
            return Local.from_str(value)
        elif Remote.accepts(value):
            return Remote.from_str(value)
        else:
            msg = f'Cannot parse source from string: {value!r}'
            raise ValueError(msg)

    @cached_property
    def extension(self):
        """File extension for the imagery in the source."""
        raise NotImplementedError

    @cached_property
    def zoom(self):
        """
        XYZ zoom level for the source.
        Our model performs best with a zoom of at least 19.
        """
        # todo: I think the difference between cfg zoom and source zoom is that cfg zoom is a suggestion,
        #   while remote zoom will be the most suitable zoom offered by the server
        raise NotImplementedError

    @cached_property
    def dimension(self):
        """Default dimension of the remote grid, e.g. 256 pixels."""
        return 256

    @classmethod
    def accepts(cls, str) -> bool:
        raise NotImplementedError
