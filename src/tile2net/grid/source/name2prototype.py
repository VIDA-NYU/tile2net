from __future__ import annotations

import copy
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
        # problem is, this approach doesn't cache, so each time, we have to regenerate.
        if self._manifest is not None:
            return self._manifest
        out = self.__class__()
        out.__dict__.update(owner._name2prototype.__dict__)
        out.__dict__.update(self.__dict__)
        file = cfg.download.servers
        name2prototype = owner.from_yaml(file)
        for name, prototype in name2prototype.items():
            # todo: somehow, this is happening regardless; why?
            from tile2net.grid.source.remote import Remote
            # if name in out.data:
                # msg = (
                #     f'Name "{name}" found both in the defined {Remote.__qualname__} '
                #     f'subclass {prototype.__class__.__qualname__} as well as in the {file} file.\n\t'
                #     'Set `enabled=False` for the subclass or the YAML entry.'
                # )
                # logger.warning(msg)
            out[name] = prototype
        # setattr(Remote, self.__name__, out)
        self._manifest = out
        return out

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        self._manifest = None

