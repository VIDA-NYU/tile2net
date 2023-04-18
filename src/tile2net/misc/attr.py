from __future__ import annotations

import functools
import os
from typing import Callable, Type
from weakref import WeakKeyDictionary

from pandas.core.generic import NDFrame


__all__ = ['attr', 'subframe']

class AttrMeta(type):
    def __get__(cls, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: attr, instance, owner):
            # saves if not on disk, loads if on disk
            self.frame = instance
            self.Frame = type(instance)
            if instance is None:
                return self

            if not self:
                res = self.fget(instance)
                self.__set__(instance, res)

            res = func(self, instance, owner)
            return res

        return wrapper

    def __set__(cls, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: attr, instance: NDFrame, value):
            self.frame = instance
            self.Frame = type(instance)
            func(self, instance, value)
            instance.attrs[self.name] = value

        return wrapper

    def __delete__(cls, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: attr, instance: NDFrame):
            self.frame = instance
            self.Frame = type(instance)
            func(self, instance)
            del instance.attrs[self.name]

        return wrapper

    def __new__(cls, name, bases, local):
        for wrapper in '__get__ __set__ __delete__'.split():
            # wrapping this way so that wrapper is outermost
            if wrapper not in local:
                continue
            func = local[wrapper]
            decorate = cls.__dict__[wrapper]
            local[wrapper] = decorate(cls, func)
        return super().__new__(cls, name, bases, local)

class attr(metaclass=AttrMeta):
    frame: NDFrame
    Frame: Type[NDFrame]

    def __init__(self, func=None, *, pickle=False, **kwargs):
        """
        Caches an attribute across Frame instances.

        :param func: Function or property to call to get the subframe.
        :param pickle: Pickle the result of the function. Owner instance must implement __fspath__
        """
        if hasattr(func, '__get__'):
            func = func.__get__
        self.fget = func
        self.pickle = pickle

    def __fspath__(self):
        return os.path.join(
            self.frame.__fspath__(),
            self.name + '.pkl'
        )

    def __set__(self, instance: NDFrame, value):
        instance.attrs[self.name] = value

    def __get__(self, instance: NDFrame, owner):
        return instance.attrs[self.name]

    def __delete__(self, instance: NDFrame):
        del instance.attrs[self.name]

    def __call__(self, func):
        if hasattr(func, '__get__'):
            func = func.__get__
        self.fget = func
        return self

    def __set_name__(self, owner, name):
        self.name = name
        functools.update_wrapper(self, self.fget)

    def __repr__(self):
        return f'<{self.__class__.__qualname__} {self.name} at {hex(id(self))}>'

    def __hash__(self):
        return hash(id(self.frame))

    def __eq__(self, other):
        return self.frame is other.frame

    def __bool__(self):
        return self.name in self.frame.attrs

class subframe(attr):

    def __init__(self, func=None, *, pickle=False, **kwargs):
        """
        Caches a Frame or Series as an attribute of the parent frame, preserving it across
        instances, and reindexing it to the parent frame's index.

        :param func: Function or property to call to get the subframe.
        :param pickle: Pickle the result of the function.
            Owner instance must implement __fspath__.
        :param kwargs: Parameters to pass to NDFrame.reindex
        """
        super().__init__(func, pickle=pickle, **kwargs)
        # self.kwargs = kwargs
        self.aligned: WeakKeyDictionary[..., set[str]] = WeakKeyDictionary()

    def __get__(self, instance, owner) -> NDFrame:
        res: NDFrame = super().__get__(instance, owner)
        aligned = self.aligned.setdefault(self, set())
        if self not in aligned:
            # params = inspect.signature(self.fget).parameters
            # kwargs = {
            #     k: v
            #     for k, v in self.kwargs.items()
            #     if k in params
            # }
            # res = res.reindex(instance.index, **kwargs)
            res = res.reindex(instance.index)
            self.__set__(instance, res)
            aligned.add(self.name)
        return res

    def __set_name__(self, owner, name):
        def fget(func):
            @functools.wraps(func)
            def wrapper(instance):
                if (
                        self.pickle
                        and os.path.exists(self.__fspath__())
                ):
                    res = self.Frame.read_pickle(self.__fspath__())
                else:
                    fget = func
                    if hasattr(fget, '__get__'):
                        fget = fget.__get__(instance, type(instance))
                    res = fget(instance)
                    if self.pickle:
                        os.makedirs(
                            os.path.dirname(self.__fspath__()),
                            exist_ok=True
                        )
                        res.to_pickle(self.__fspath__())
                if self.pickle:
                    res.unlink = functools.partial(os.unlink, self.__fspath__())

                return res

            return wrapper

        self.fget = fget(self.fget)
        super().__set_name__(owner, name)

if __name__ == '__main__':
    from pandas import DataFrame


    class Test(DataFrame):
        @attr
        @property
        def scalar(self):
            print('SCALAR')
            return 1

        #
        @subframe
        @property
        def series(self):
            print('SERIES')
            return self.iloc[:, 0]

        @subframe
        @property
        def frame(self):
            print('FRAME')
            return self.copy()

        @property
        def _constructor(self) -> Callable[..., Test]:
            return type(self)

        @property
        def prop(self):
            return True

    test = Test({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
    })
    test.scalar = 2
    test.series
    test.frame
    sub = test.iloc[:2]
    sub.scalar
    sub.series
    sub.frame
    pass
