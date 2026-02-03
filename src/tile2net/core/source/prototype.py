from __future__ import annotations

from collections import UserDict
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from .remote import Remote


class Prototype(
    UserDict
):
    data: dict[type[Remote], Remote]

    def __getitem__(self, key: str) -> Remote:
        return super().__getitem__(key)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(self, dict=None, /, **kwargs):
        if callable(dict):
            dict = None
        super().__init__(dict, **kwargs)

    @overload
    def __get__[T](
            self,
            instance,
            owner: type[T]
    ) -> T:
        ...

    @overload
    def __get__[T](
            self,
            instance: T,
            owner
    ) -> T:
        ...

    def __get__(
            self,
            instance: Remote,
            owner: type[Remote]
    ):
        """Returns the prototype instance for the accessing class (owner)."""
        from tile2net.core.source.remote import Remote
        if (
            isinstance(instance, Remote)
            and self.__name__ in instance.__dict__
        ):
            return instance.__dict__[self.__name__]
        if owner not in self:
            self[owner] = owner()
        return self.data[owner]

    def __set__(
            self,
            instance: Remote,
            value,
    ):
        instance.__dict__[self.__name__] = value

    def __delete__(
            self,
            instance: Remote,
    ):
        if self.__name__ in instance.__dict__:
            del instance.__dict__[self.__name__]



