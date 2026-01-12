from __future__ import annotations

from collections import UserDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .remote import Remote


class Catalog(
    UserDict
):
    data: dict[str, Remote]

    def __getitem__(self, key: str) -> Remote:
        return super().__getitem__(key)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(self, dict=None, /, **kwargs):
        if callable(dict):
            dict = None
        super().__init__(dict, **kwargs)

    def __repr__(self) -> str:
        lines = (f'    "{k}": {v!r}' for k, v in self.data.items())
        return "{\n" + ",\n".join(lines) + "\n}"