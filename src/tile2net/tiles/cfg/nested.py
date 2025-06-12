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
        self._cfg = instance
        self._Cfg = owner
    else:
        self._cfg = instance._cfg
        self._Cfg = instance._Cfg

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
    def _cfg(self) -> Optional[Cfg]:
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
                # or self.instance.instance is None
            or not isinstance(self.instance, Nested)
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

    # def __getattr__(self, key: str) -> Any:
    #     if key.isupper():
    #         key = key.lower()
    #
    #     attr_err: AttributeError | None = None
    #     try:
    #         return object.__getattribute__(self, key)
    #     except AttributeError as e:
    #         attr_err = e
    #
    #     if self._cfg is not None:
    #         trace_key = f"{self._trace}.{key}"
    #         if trace_key in self._cfg:
    #             return self._cfg[trace_key]
    #
    #     raise attr_err if attr_err is not None else AttributeError(key)
    #
    # def __setattr__(self, key: str, value: Any) -> None:
    #     cls = self.__class__
    #
    #     # No owner yet (during __set_name__) → default behaviour
    #     # if self.owner is None:
    #     if object.__getattribute__(self, 'owner') is None:
    #         return object.__setattr__(self, key, value)
    #
    #     # Normalize ALL-CAPS names to lowercase
    #     if key.isupper():
    #         key = key.lower()
    #
    #     # 1. internal/descriptor attributes
    #     if key.startswith('_') or hasattr(cls, key):
    #         return object.__setattr__(self, key, value)
    #
    #     if self._cfg is not None:
    #         # 2. fallback to cfg dict
    #         trace_key = f"{self._trace}.{key}"
    #         self._cfg[trace_key] = value
    #
    # def __delattr__(self, key: str) -> None:
    #     cls = self.__class__
    #
    #     # Normalize ALL-CAPS names to lowercase
    #     if key.isupper():
    #         key = key.lower()
    #
    #     # 1. internal/descriptor attributes
    #     if key.startswith('_') or hasattr(cls, key):
    #         return object.__delattr__(self, key)
    #
    #     # 2. try to remove from cfg dict
    #     # if self.cfg is not None:
    #     if isinstance(self._cfg, Cfg):
    #         trace_key = f"{self._trace}.{key}"
    #         if trace_key in self._cfg:
    #             del self._cfg[trace_key]
    #             return
    #
    #     raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}")

    def __getattr__(self, key: str) -> Any:
        try:
            return object.__getattribute__(self, key)
        except AttributeError as attr_err:
            low_key = key.lower()
            if low_key != key:
                try:
                    return object.__getattribute__(self, low_key)
                except AttributeError:
                    pass

            if self._cfg is not None:
                for k in (key, low_key):
                    trace_key = f"{self._trace}.{k}"
                    if trace_key in self._cfg:
                        return self._cfg[trace_key]

            raise attr_err

    def __setattr__(self, key: str, value: Any) -> None:
        cls = self.__class__

        # Owner not yet assigned → default behaviour
        if object.__getattribute__(self, 'owner') is None:
            return object.__setattr__(self, key, value)

        # 1. internal/descriptor attributes
        if key.startswith('_') or hasattr(cls, key):
            return object.__setattr__(self, key, value)

        # 2. first attempt with original key
        # if self._cfg is not None:
        #     trace_key = f"{self._trace}.{key}"
        #     self._cfg[trace_key] = value
        #     return

        # 3. last-resort lowercase key
        low_key = key.lower()
        if low_key != key and self._cfg is not None:
            trace_key = f"{self._trace}.{low_key}"
            self._cfg[trace_key] = value
            return

        # Fallback to default setattr if cfg absent
        object.__setattr__(self, key, value)

    def __delattr__(self, key: str) -> None:
        cls = self.__class__

        # 1. internal/descriptor attributes
        if key.startswith('_') or hasattr(cls, key):
            return object.__delattr__(self, key)

        # 2. try original and lowercase keys in cfg
        if isinstance(self._cfg, Cfg):
            for k in (key, key.lower()):
                trace_key = f"{self._trace}.{k}"
                if trace_key in self._cfg:
                    del self._cfg[trace_key]
                    return

        raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}")

    def __getattr__(self, key: str) -> Any:
        KEY = key
        key = KEY
        if (
                key.startswith('_')
                or key == 'data'
        ):
            return object.__getattribute__(self, key)
        key = key.lower()
        if hasattr(self.__class__, key):
            return object.__getattribute__(self, key)
        if self._trace:
            trace = f'{self._trace}.{key}'
        else:
            trace = key

        try:
            return self._cfg[trace]
        except KeyError:
            ...

        try:
            attrs = trace.split('.')
            obj = self._cfg
            for attr in attrs[:-1]:
                obj = object.__getattribute__(obj, attr)
            result = object.__getattribute__(obj, attrs[-1])
            return result
        except AttributeError as e:
            msg = f'{self.__class__.__name__} has no attribute {key!r} (trace: {trace})'
            raise AttributeError(msg) from e

    def __setattr__(self, key: str, value: Any) -> None:
        KEY = key
        key = KEY
        if (
                key.startswith('_')
                or key == 'data'
        ):
            return object.__setattr__(self, key, value)
        key = key.lower()
        if hasattr(self.__class__, key):
            return object.__setattr__(self, key, value)
        if self._trace:
            trace = f'{self._trace}.{key}'
        else:
            trace = key

        try:
            self._cfg[trace] = value
        except KeyError as e:
            msg = f'{self.__class__.__name__} has no attribute {key!r} (trace: {trace})'
            raise AttributeError(msg) from e

        try:
            attrs = trace.split('.')
            obj = self._cfg
            for attr in attrs[:-1]:
                obj = object.__getattribute__(obj, attr)
            object.__setattr__(obj, attrs[-1], value)
        except AttributeError as e:
            msg = f'{self.__class__.__name__} has no attribute {key!r} (trace: {trace})'
            raise AttributeError(msg) from e

    def __delattr__(self, key: str) -> None:
        KEY = key
        key = KEY
        if (
                key.startswith('_')
                or key == 'data'
        ):
            return object.__delattr__(self, key)
        key = key.lower()
        if hasattr(self.__class__, key):
            return object.__delattr__(self, key)
        if self._trace:
            trace = f'{self._trace}.{key}'
        else:
            trace = key

        try:
            del self._cfg[trace]
        except KeyError as e:
            ...

        try:
            attrs = trace.split('.')
            obj = self._cfg
            for attr in attrs[:-1]:
                obj = object.__getattribute__(obj, attr)
            object.__delattr__(obj, attrs[-1])
        except AttributeError as e:
            msg = f'{self.__class__.__name__} has no attribute {key!r} (trace: {trace})'
            raise AttributeError(msg) from e

