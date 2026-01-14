from __future__ import annotations

import copy
from functools import *
from types import TracebackType
from typing import *

from tile2net.grid.util import returns_or_assigns
from .wrapper import Wrapper

if False:
    from tile2net.grid.grid.grid import Grid

TGrid = TypeVar('TGrid', covariant=True)


class namespace(
):
    __wrapped__ = None
    __name__ = None
    instance: object

    def _get(
            self,
            instance,
            owner
    ) -> Self:
        self.instance = instance
        if instance is None:
            self.wrapper = None
        elif isinstance(instance, Wrapper):
            self.wrapper = instance
        elif isinstance(instance, namespace):
            self.wrapper = instance.wrapper
        else:
            raise TypeError(instance)
        return copy.copy(self)

    locals().update(__get__=_get)

    def __init__(
            self,
            func=None,
            *args,
            **kwargs
    ):
        if (
                callable(func)
                and returns_or_assigns(func)
        ):
            update_wrapper(self, func)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __delete__(
            self,
            instance: Grid,
    ):
        del instance.__dict__[self.__name__]

    def __set__(
            self,
            instance: Grid,
            value: namespace,
    ):
        instance.__dict__[self.__name__] = value
        value.instance = instance

    def __enter__(self):
        """
        Supports using namespace as a truthy context manager,
        allowing for creative and convenient uses. For example:
        >>> with self.file:
        >>>     static = self.file.Static
        The retrieval of `static` uses the truthy context to behave differently.
        Normally, it is configured to start downloading the static files for user
        convenience. With this context manager, it allows us to check if `self.file`
        to temporarily disable the auto-download, getting the paths without downloading.
        """
        self.instance.__dict__[self.__name__] = True

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc: Optional[BaseException],
            tb: Optional[TracebackType],
    ):
        try:
            del self.instance.__dict__[self.__name__]
        except KeyError:
            ...
        return False

    def __bool__(self):
        # seee __enter__/__exit__
        return (
            self.instance.__dict__
            .get(self.__name__, False)
        )
