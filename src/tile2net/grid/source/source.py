from __future__ import annotations

from abc import abstractmethod, ABC
from functools import cached_property
from typing import *

from shapely import box

from tile2net.logger import logger

if not TYPE_CHECKING:
    from ..ingrid import InGrid

from tile2net.grid.cfg.logger import logger
from tile2net.grid.source.exceptions import SourceParseError


class Source(
    ABC
):
    @property
    def files(self):
        raise NotImplementedError

    def __get__(
            self: Source,
            instance: InGrid,
            owner: type[InGrid],
    ) -> Source:
        """Return the remote object for the grid instance."""
        key = self.__name__
        cache = instance.__dict__
        if key not in cache:
            msg = (
                f'Source has not been set. Defaulting to a remote source that '
                f'best fits the the provided location.'
            )
            logger.info(msg)
            bounds = instance.geometry.total_bounds
            item = box(*bounds).centroid
            from .remote import Remote
            out = Remote.from_location(item)
            cache[key] = out
        return cache[key]


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
    def from_inferred(cls, value) -> Self:
        """ Infer the appropriate Source (Local or Remote) from various input types. """
        from .remote import Remote
        from .local import Local

        try:
            return Local.from_inferred(value)
        except SourceParseError:
            pass

        try:
            return Remote.from_inferred(value)
        except SourceParseError:
            pass

        msg = f'Cannot infer source from value: {value!r}'
        raise ValueError(msg)

    @cached_property
    def extension(self):
        """File extension for the imagery in the source."""
        raise NotImplementedError

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
