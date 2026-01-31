from __future__ import annotations

import copy
from typing import *

import geopandas as gpd
import shapely

from tile2net.grid.basegrid.basegrid import BaseGrid
from tile2net.grid.cfg.logger import logger
from tile2net.grid.dir.dir import Dir
from tile2net.grid.frame.weak import weak
from tile2net.grid.source.exceptions import SourceParseError
from tile2net.logger import logger

if TYPE_CHECKING:
    from ..grid import Grid


class Source:
    extension: str
    """File extension for the imagery in the source."""

    zoom: int
    """XYZ zoom level for the source.
    Our model performs best with a zoom of at least 19.
    """

    dimension: int
    """Default dimension of the remote grid, e.g. 256 pixels."""

    @weak.property
    def grid(self) -> Optional[Grid]:
        """The Grid instance this source is attached to."""
        return None

    def _get(
            self,
            instance: Grid,
            owner: type[Grid],
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
        if isinstance(out, Source):
            out.grid = instance
        if instance is None:
            out.basegrid = None
        elif isinstance(instance, BaseGrid):
            out.basegrid = instance
        elif isinstance(instance, Dir):
            out.basegrid = instance.basegrid
        else:
            raise TypeError(instance)
        out.instance = instance
        return out

    locals().update(__get__=_get)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(
            self,
            instance: Grid,
            value: Union[Source, str, None],
    ):
        if value is None:
            value = self.from_grid(instance)
        if isinstance(value, Source):
            value = copy.copy(value)
        elif isinstance(value, str):
            value = self.from_inferred(value)
        elif value is False:
            ...
        else:
            msg = (
                f'Cannot set source with value of type {type(value)}. '
                f'Expected either a Source or a str.'
            )
            raise TypeError(msg)
        value.__dict__ = self.__dict__ | value.__dict__
        instance.__dict__[self.__name__] = value

    def __delete__(self, instance: Grid):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            ...

    @classmethod
    def from_grid(cls, value: BaseGrid) -> Self:
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

        if isinstance(value, Source):
            return copy.copy(value)

        try:
            out = Local.from_inferred(value)
        except SourceParseError:
            ...
        else:
            return out
        try:
            out = Remote.from_inferred(value)
        except SourceParseError:
            ...
        else:
            if not out.server:
                msg = f'Remote source must have a server defined: {value!r}'
                raise ValueError(msg)
            return out

        msg = f'Cannot infer source from value: {value!r}'
        raise ValueError(msg)

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        """Empty init just to allow use as a decorator."""
