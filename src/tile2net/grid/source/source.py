from __future__ import annotations
from typing import *

import copy
from tile2net.grid.frame.weak import weak
from abc import abstractmethod, ABC
from functools import cached_property
from typing import *

import geopandas as gpd
import shapely

from tile2net.grid.grid.grid import Grid
from tile2net.logger import logger

if not TYPE_CHECKING:
    from ..ingrid import InGrid

from tile2net.grid.cfg.logger import logger
from tile2net.grid.source.exceptions import SourceParseError


class Source(
    ABC
):
    @cached_property
    @abstractmethod
    def extension(self):
        """File extension for the imagery in the source."""

    @cached_property
    @abstractmethod
    def zoom(self):
        """
        XYZ zoom level for the source.
        Our model performs best with a zoom of at least 19.
        """

    @cached_property
    @abstractmethod
    def dimension(self):
        """Default dimension of the remote grid, e.g. 256 pixels."""

    @weak.property
    @abstractmethod
    def ingrid(self) -> InGrid:
        """The InGrid instance this source is attached to."""

    def _get(
            self,
            instance: InGrid,
            owner: type[InGrid],
    ) -> Self:
        """Return the remote object for the grid instance."""
        if instance is None:
            return self
        key = self.__name__
        cache = instance.__dict__
        if key not in cache:
            source = instance.cfg.source
            msg = 'Source has not been set. '
            if not source:
                msg += 'Defaulting to a remote source that best fits the the provided location.'
            else:
                msg += f'Setting source from cfg: {source!r}'
            logger.info(msg)
            self.__set__(instance, source)

        out: Self = cache[key]
        out.ingrid = instance
        return out

    locals().update(__get__=_get)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(
            self,
            instance: InGrid,
            value: Union[Source, str, None],
    ):
        if value is None:
            value = self.from_grid(instance)
        if isinstance(value, Source):
            value = copy.copy(value)
        elif isinstance(value, str):
            value = self.from_inferred(value)
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
    def from_grid(cls, value: Grid) -> Self:
        from .remote import Remote
        bbox_geom = shapely.geometry.box(*value.frame.total_bounds)
        out = (
            gpd.GeoSeries([bbox_geom], crs=value.frame.crs)
            .to_crs(4326)
            .pipe(Remote.from_inferred)
        )
        return out

    @classmethod
    def from_inferred(cls, value) -> Self:
        """ Infer the appropriate Source (Local or Remote) from various input types. """
        from .remote import Remote
        from .local import Local

        try:
            return Local.from_inferred(value)
        except SourceParseError:
            ...

        try:
            return Remote.from_inferred(value)
        except SourceParseError:
            ...

        msg = f'Cannot infer source from value: {value!r}'
        raise ValueError(msg)

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        """Empty init just to allow use as a decorator."""
