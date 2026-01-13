from __future__ import annotations

from collections import UserDict

from typing import TYPE_CHECKING, Self, overload

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



    def __get__(
            self,
            instance,
            owner
    ):
        """Returns the prototype instance for the accessing class (owner)."""
        if owner not in self:
            self[owner] = owner()
        return self[owner]
