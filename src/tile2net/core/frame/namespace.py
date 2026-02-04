from __future__ import annotations

import copy
from functools import *
from types import TracebackType
from typing import *

from .wrapper import Wrapper

if TYPE_CHECKING:
    from tile2net.core.grid.grid import Grid


class namespace:
    __wrapped__ = None
    __name__ = None
    instance: namespace = None
    wrapper: Wrapper = None

    @overload
    def _get[T](
            self: T,
            instance,
            owner
    ) -> T:
        ...

    @overload
    def _get[T](
            self: T,
            instance,
            owner
    ) -> T:
        ...

    def _get(
            self,
            instance: namespace,
            owner: type[namespace],
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
        from tile2net.core.util import returns_or_assigns
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
        # see __enter__/__exit__
        return (
            self.instance.__dict__
            .get(self.__name__, False)
        )

    @property
    def _trace(self) -> str:
        """String which replicates the attribute access chain. """
        instance = self.instance
        if instance is None:
            return ''
        key = f'{self.__name__}._trace'
        if key not in instance.__dict__:
            if (
                isinstance(instance, namespace)
                and instance._trace
            ):
                trace = f'{instance._trace}.{self.__name__}'
            else:
                trace = self.__name__
            instance.__dict__[key] = trace
        return instance.__dict__[key]
