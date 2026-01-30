from __future__ import annotations

import copy
from collections import UserDict
from functools import *
from typing import *

if TYPE_CHECKING:
    from .cfg import Cfg


class _Nested(UserDict[str, 'Nested']):
    data: dict[
        type[Nested],
        dict[str, Nested]
    ]

    def _get(
            self,
            instance,
            owner: type[Nested]
    ) -> dict[str, Nested]:
        cache = self.data

        if instance is None:
            if owner not in cache:
                cache[owner] = {
                    name: nested
                    for base in reversed(owner.__bases__)
                    if issubclass(base, Nested)
                    for name, nested in base._nested.items()
                }
                for name in cache[owner]:
                    assert isinstance(name, str)

            return cache[owner]
        else:
            if self.__name__ not in instance.__dict__:
                instance.__dict__[self.__name__] = {
                    name: getattr(instance, name)
                    for name, nested in owner._nested.items()
                }
                for name in instance.__dict__[self.__name__]:
                    assert isinstance(name, str)
            return instance.__dict__[self.__name__]

    locals().update(__get__=_get)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(
            self,
            instance: Nested,
            value,
    ):
        instance.__dict__[self.__name__] = value

    def __delete__(
            self,
            instance,
    ):
        if self.__name__ in instance.__dict__:
            del instance.__dict__[self.__name__]


class Nested:
    instance: Nested = None
    owner: Type[Nested] = None
    _nested = _Nested()
    _owner2instance: dict[Type[Nested], Nested]

    def _get(
            self,
            instance: Nested,
            owner: Type[Nested],
    ) -> Self:
        if instance is None:
            # from owner
            if owner in self._owner2instance:
                out = self._owner2instance[owner]
            else:
                out = copy.copy(self)
        else:
            # from instance
            key = self.__name__
            if key in instance.__dict__:
                out = instance.__dict__[key]
            else:
                out = copy.copy(self)
                instance.__dict__[key] = out

        out.instance = instance
        out.owner = owner

        if isinstance(instance, Nested):
            out._cfg = instance._cfg
            out._Cfg = instance._Cfg
        elif issubclass(owner, Nested):
            out._cfg = instance
            out._Cfg = owner
        else:
            raise NotImplementedError

        return out

    locals().update(__get__=_get)

    @cached_property
    def _cfg(self) -> Optional[Cfg]:
        return None

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    # @cached_property
    # def _trace(self):
    #     if (
    #             self.instance is None
    #             or not isinstance(self.instance, Nested)
    #     ) or (
    #             isinstance(self.instance, Nested)
    #             and not self.instance._trace
    #     ):
    #         out = self.__name__
    #     elif (
    #         isinstance(self.instance, Nested)
    #         and self.instance._trace
    #     ):
    #         out = f'{self.instance._trace}.{self.__name__}'
    #     elif isinstance(self.instance, Nested):
    #         out = self.__name__
    #     else:
    #         msg = (
    #             f'Cannot determine trace for {self.__name__} in '
    #             f'{self.owner} with {self.instance=}'
    #         )
    #         raise ValueError(msg)
    #     return out

    @property
    def _trace(self):
        # problem is, the object is already stored, instead of trace
        from .cfg import Cfg
        instance = self.instance
        # name = self.__name__
        name = f'{self.__name__}'
        key = f'{self.__name__}.trace'
        if isinstance(instance, Cfg):
            out = name
        elif isinstance(instance, Nested):
            if key not in instance.__dict__:
                instance.__dict__[key] = f'{instance._trace}.{name}'
            out = instance.__dict__[key]
        elif instance is None:
            out = name
        else:
            msg = (
                f'Cannot determine trace for {name} in '
                f'{self.owner} with {instance=}'
            )
            raise ValueError(msg)
        return out


    def __set_name__(
            self,
            owner: type[Nested],
            name
    ):
        self.__name__ = name
        self.owner = owner
        if issubclass(owner, Nested):
            owner._nested[name] = self
        self._owner2instance = {}

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

        if self._cfg is not None:
            try:
                return self._cfg._lookup(trace)
            except KeyError:
                ...

        attrs = trace.split('.')
        obj = self._Cfg
        for attr in attrs[:-1]:
            obj = object.__getattribute__(obj, attr)
        result = object.__getattribute__(obj, attrs[-1])
        return result

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

