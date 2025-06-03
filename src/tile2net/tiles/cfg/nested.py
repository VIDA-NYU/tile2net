from __future__ import annotations

from functools import *
from typing import *

if False:
    from .cfg import Cfg
    from ..tiles import Tiles


def __get__(
        self: Nested,
        instance: Nested,
        owner: Type[Nested],
):
    """"""
    from .cfg import Cfg
    self.instance = instance
    self.owner = owner
    if issubclass(owner, Cfg):
        self.cfg = instance
        self.Cfg = owner
    else:
        self.cfg = instance.cfg
        self.Cfg = instance.Cfg

    return self


class Nested(

):
    instance: Nested = None
    owner: Type[Nested] = None
    _nested: dict[str, Nested]

    locals().update(
        __get__=__get__
    )

    @cached_property
    def tiles(self) -> Tiles:
        return None

    @cached_property
    def cfg(self) -> Optional[Cfg]:
        return None

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    @cached_property
    def _trace(self):
        if (
                self.instance is None
                or self.instance.instance is None
        ):
            return self.__name__
        elif isinstance(self.instance, Nested):
            return f'{self.instance._trace}.{self.__name__}'
        else:
            msg = (
                f'Cannot determine trace for {self.__name__} in '
                f'{self.owner} with {self.instance=}'
            )
            raise ValueError(msg)

    def __set_name__(
            self,
            owner: type[Nested],
            name
    ):
        self.__name__ = name
        self.owner = owner
        if issubclass(owner, Nested):
            if '_nested' not in owner.__dict__:
                owner._nested = {}
            owner._nested[name] = self

    def __getattr__(self, key: str) -> Any:
        # Normalize ALL-CAPS names to lowercase
        if key.isupper():
            key = key.lower()

        # 1. descriptor / attribute on the instance or class
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            pass

        # 2. fallback to cfg dict
        if self.cfg is not None:
            trace_key = f"{self._trace}.{key}"
            if trace_key in self.cfg:
                return self.cfg[trace_key]

        raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}")

    def __setattr__(self, key: str, value: Any) -> None:
        cls = self.__class__

        # No owner yet (during __set_name__) → default behaviour
        # if self.owner is None:
        if object.__getattribute__(self, 'owner') is None:
            return object.__setattr__(self, key, value)

        # Normalize ALL-CAPS names to lowercase
        if key.isupper():
            key = key.lower()

        # 1. internal/descriptor attributes
        if key.startswith('_') or hasattr(cls, key):
            return object.__setattr__(self, key, value)

        if self.cfg is not None:
            # 2. fallback to cfg dict
            trace_key = f"{self._trace}.{key}"
            self.cfg[trace_key] = value

    def __delattr__(self, key: str) -> None:
        cls = self.__class__

        # Normalize ALL-CAPS names to lowercase
        if key.isupper():
            key = key.lower()

        # 1. internal/descriptor attributes
        if key.startswith('_') or hasattr(cls, key):
            return object.__delattr__(self, key)

        # 2. try to remove from cfg dict
        # if self.cfg is not None:
        if isinstance(self.cfg, Cfg):
            trace_key = f"{self._trace}.{key}"
            if trace_key in self.cfg:
                del self.cfg[trace_key]
                return

        raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}")
