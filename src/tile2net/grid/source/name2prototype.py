from __future__ import annotations

from collections import UserDict
from typing import *
from typing import TYPE_CHECKING

from tile2net.grid.cfg import cfg
from tile2net.logger import logger

if TYPE_CHECKING:
    from .remote import Remote


class _Name2Prototype(
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


class Name2Prototype(
    _Name2Prototype
):
    def __get__(
            self,
            instance,
            owner: Remote
    ) -> Self:
        from tile2net.grid.source.remote import Remote
        out = self.__class__(owner._name2prototype.copy())
        file = cfg.download.servers
        name2prototype = owner.from_yaml(file)
        for name, prototype in name2prototype.items():
            if name in out.data:
                msg = (
                    f'Name {name} found both in the defined {Remote.__qualname__} '
                    f'subclass {prototype.__class__.__qualname__} as well as in the {file} file.\n',
                    'Set `enabled=False` for the subclass or the YAML entry.'
                )
                logger.warning(msg)
        setattr(Remote, self.__name__, out)
        return out
