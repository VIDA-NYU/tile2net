from __future__ import annotations

import copy
import inspect
from collections import UserDict
from pathlib import Path
from typing import *
from typing import TYPE_CHECKING

from tile2net.logger import logger

if TYPE_CHECKING:
    from .remote import Remote
    from .source import Source


class _Name2Prototype(
    UserDict
):
    data: dict[
        type[Remote],
        dict[str, Remote]
    ]

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

    def __get__(
            self,
            instance,
            owner: type[Remote]
    ) -> Self:
        base = owner.base
        if base in self.data:
            return self.data[base]
        self._owner = owner
        out = {}
        self.data[base] = out
        return out


class Name2Prototype(
    _Name2Prototype
):
    data: dict[str, Source]
    _cache: dict[type[Source], Self]
    _owner = None

    def __getitem__(self, key: str) -> Source:
        return super().__getitem__(key)

    def __set_name__(self, owner: type[Source], name):
        self.__name__ = name
        self._cache: dict[type[Source], Self] = dict()

    def __repr__(self) -> str:
        lines = [f"{self._owner.__qualname__}.{self.__name__}:"]

        for key, source in sorted(self.data.items()):
            lines.append(f"  {key}:")

            attrs = {
                attr: getattr(source, attr, None)
                for attr in ['desc', 'server', 'base']
            }

            for attr, val in attrs.items():
                if not val:
                    continue

                display_val = (
                    val.__name__
                    if isinstance(val, type)
                    else val
                )
                lines.append(f"    {attr}: {display_val}")

        return "\n".join(lines)

    def __init__(
            self,
            dict: Union[
                Callable[[type[Remote]], dict[str, Remote]],
                None,
                dict,
            ] = None,
            /,
            **kwargs
    ):
        if callable(dict):
            super().__init__(**kwargs)
            self.__wrapped__ = dict
        else:
            super().__init__(dict, **kwargs)

    def __get__(
            self,
            instance: Remote,
            owner: type[Remote]
    ) -> Self:
        base = owner
        if base in self._cache:
            return self._cache[base]
        self._owner = owner

        out = copy.copy(self)
        out.data.update(base._name2prototype)
        self._cache[base] = out

        # 1. Inherit from derived
        out.update({
            name: prototype
            for derived in base.derived
            for name, prototype in derived.name2prototype.items()
        })

        # 2. Inherit from wrapped
        data = self.__wrapped__(owner)
        if data:
            for name, prototype in data.items():
                if name in out:
                    msg = (
                        f'Name "{name}" found both in the inherited {base.__qualname__} '
                        f'prototypes and the wrapped definition.\n\t'
                        'The wrapped value will overwrite the inherited value.'
                    )
                    logger.warning(msg)
                out[name] = prototype

        # 3. Inherit from yaml
        yaml = self.yaml(owner)
        if yaml:
            name2prototype = base.from_yaml(yaml)
            for name, prototype in name2prototype.items():
                if name in out:
                    msg = (
                        f'Name "{name}" found both in the defined {base.__qualname__} '
                        f'subclass {prototype.__class__.__qualname__} as well as in the {yaml} file.\n\t'
                        'Set `enabled=False` for the subclass or the YAML entry.'
                    )
                    logger.warning(msg)
                out[name] = prototype

        return out

    def yaml(self, cls: type[Remote]) -> Optional[str]:
        """Returns filepath to associated yaml file if it exists."""
        if cls.base is not cls:
            return None
        try:
            out = inspect.getfile(cls)
        except TypeError:
            logger.warning(f'Unable to determine a file-path for built-in {cls.__qualname__}.')
            return None
        out = (
            Path(out)
            .with_suffix('.yaml')
        )
        if not out.exists():
            return None
        out = str(out)
        return out
