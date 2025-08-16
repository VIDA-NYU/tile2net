from __future__ import annotations

from functools import *
from typing import *

if False:
    from .cfg import Cfg
    from ..grid import Grid




class Nested(

):
    instance: Nested = None
    owner: Type[Nested] = None
    _nested: dict[str, Nested]

    def _get(
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

    locals().update(
        __get__=_get
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

        # try:
        #     return self._cfg[trace]
        # except KeyError:
        #     ...

        # try if it's cached
        # try:
        #     return self._cfg._lookup(trace)
        # except KeyError as e:
        #     ...
        try:
            return self._cfg._lookup(trace)
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

        # try:
        #     attrs = trace.split('.')
        #     obj = self._cfg
        #     for attr in attrs[:-1]:
        #         obj = object.__getattribute__(obj, attr)
        #     object.__setattr__(obj, attrs[-1], value)
        # except AttributeError as e:
        #     msg = f'{self.__class__.__name__} has no attribute {key!r} (trace: {trace})'
        #     raise AttributeError(msg) from e

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
