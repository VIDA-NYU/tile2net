from __future__ import annotations

from collections import UserDict
from typing import *

if TYPE_CHECKING:
    from .remote import Remote


class Base:
    """
    A descriptor that resolves the 'Base' class for any given Remote subclass.
    Uses a static class-level dictionary for O(1) caching and registration.
    """
    data: dict[
        type[Remote],
        type[Remote]
    ] = {}

    def __set_name__(self, owner, name):
        self.__name__ = name

    @overload
    def __get__[T](
            self,
            instance,
            owner: type[T]
    ) -> type[T]:
        ...

    @overload
    def __get__[T](
            self,
            instance: T,
            owner
    ) -> type[T]:
        ...

    def __get__(
            self,
            instance: Remote,
            owner: type[Remote]
    ) -> type[Remote]:
        if owner in self.data:
            return self.data[owner]

        for base in owner.mro():
            if base in self.data:
                base: type[Remote]
                self.data[owner] = self.data[base]
                return self.data[base]

        self.data[owner] = owner
        return owner

    def __init__(self, func=None):
        ...


class Derived(
    UserDict
):
    """
    A descriptor and registry that maintains lists of derived subclasses
    for each Base class.
    """

    def __set_name__(self, owner, name):
        self.__name__ = name

    @overload
    def __get__[T](
            self,
            instance,
            owner: type[T]
    ) -> set[type[T]]:
        ...

    @overload
    def __get__[T](
            self,
            instance: T,
            owner
    ) -> set[type[T]]:
        ...

    def __get__(
            self,
            instance: Remote,
            owner: type[Remote]
    ) -> set[type[Remote]]:
        base = owner.base
        if base not in self.data:
            self.data[base] = set()

        return self.data[base]

    def __init__(
            self,
            dict=None,
            /,
            **kwargs
    ):
        if callable(dict):
            super().__init__(**kwargs)
        else:
            super().__init__(dict, **kwargs)
