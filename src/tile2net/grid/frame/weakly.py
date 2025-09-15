from __future__ import annotations

import functools
from typing import *
import weakref


class WeakProperty:
    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self.__func__ = func

    @functools.cached_property
    def _safety(self):
        """
        Used to a single instance in the class to avoid instability during chaining.
        e.g. when accessing a.b.c, c stores b as a weakref, but a.b might not be
        concretely stored anywhere, so it is garbage collected and an error occurs
         when a.b.c tries to access b.
        """
        return f'_{self.__name__}'

    def __get__(self, instance: object, owner: type):
        key = self.__name__
        cache = instance.__dict__
        if key in cache:
            try:
                return cache[key]()
            except TypeError as e:
                raise TypeError(
                    f'weakly cached property {owner}.{self.__name__} '
                    f'returned None'
                ) from e
        result = self.__func__(instance)
        if (
                result is not None
                and not isinstance(result, weakref.ReferenceType)
        ):
            result = weakref.ref(result)
        cache[key] = result
        return result

    def __set__(self, instance, value):
        key = self.__name__
        if (
                value is not None
                and not isinstance(value, weakref.ReferenceType)
        ):
            setattr(type(instance), self._safety, value)
            value = weakref.ref(value)
        instance.__dict__[key] = value

    def __delete__(self, instance):
        del instance.__dict__[self.__name__]


class weakly:
    cached_property = WeakProperty
