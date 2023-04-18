from __future__ import annotations
import copy

import functools
import os
from _weakrefset import WeakSet
from typing import Callable, Type

import pandas as pd
import logging
import pickle
from pandas.core.generic import NDFrame


__all__ = ['attr', 'subframe']

class AttrMeta(type):
    def __get__(cls, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: attr, instance, owner):
            # saves if not on disk, loads if on disk
            self.instance = instance
            self.owner = type(instance)
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
            self.instance = instance
            self.owner = type(instance)
            func(self, instance, value)

        return wrapper

    def __delete__(cls, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: attr, instance: NDFrame):
            self.instance = instance
            self.owner = type(instance)
            func(self, instance)

        return wrapper

class attr(metaclass=AttrMeta):
    instance: NDFrame
    owner: Type[NDFrame]

    def __init__(self, func=None, *, pickle=False, log=False, **kwargs):
        """
        Caches an attribute across Frame instances.

        :param func: Function or property to call to get the subframe.
        :param pickle: Pickle the result of the function. Owner instance must implement __fspath__
        """
        if (
                not callable(func)
                and hasattr(func, '__get__')
        ):
            # func is some sort of property
            func = func.__get__
        self.fget = func
        self.pickle = pickle
        self.log = log

    def __set__(self, instance: NDFrame, value):
        instance.attrs[self.name] = value

    def __get__(self, instance: NDFrame, owner):
        return instance.attrs[self.name]

    def __delete__(self, instance: NDFrame):
        del instance.attrs[self.name]

    def __call__(self, func):
        if (
                not callable(func)
                and hasattr(func, '__get__')
        ):
            # func is some sort of property
            func = func.__get__
        self.fget = func
        return self

    def __set_name__(self, owner, name):
        self.name = name

        def fget(func):
            @functools.wraps(func)
            def wrapper(instance):
                if (
                        self.pickle
                        and os.path.exists(path := self.__fspath__())
                ):
                    if self.log:
                        logging.info(f'Loading {self.name} from {path}')
                    with open(path, 'rb') as f:
                        res = pickle.load(f)
                else:
                    fget = func
                    if (
                            not callable(fget)
                            and hasattr(fget, '__get__')
                    ):
                        # func is some sort of property
                        fget = fget.__get__
                    res = fget(instance)
                    if self.pickle:
                        path = self.__fspath__()
                        os.makedirs(
                            os.path.dirname(path),
                            exist_ok=True
                        )
                        if self.log:
                            logging.info(f'Saving {self.name} to {path}')
                        with open(path, 'wb') as f:
                            pickle.dump(res, f)
                if self.pickle:
                    res.unlink = functools.partial(os.unlink, self.__fspath__())

                return res

            return wrapper

        self.fget = fget(self.fget)
        functools.update_wrapper(self, self.fget)

    def __new__(cls, *args, **kwargs):
        cls = copy.copy(cls)
        meta = type(cls)
        # creates a copy of the class so that descriptor methods are wrapped on the outermost
        #   but the wrapper is not inherited
        for wrapper in '__get__ __set__ __delete__'.split():
            # wrapping this way so that wrapper is always outermost
            func = getattr(cls, wrapper)
            decorate = getattr(meta, wrapper)
            wrapped = decorate(meta, func)
            setattr(cls, wrapper, wrapped)
        self = super().__new__(cls)
        return self

    def __repr__(self):
        try:
            return f'<{self.__class__.__qualname__} {self.name} at {hex(id(self))}>'
        except AttributeError:
            return super().__repr__()

    def __hash__(self):
        return hash(id(self.instance))

    def __eq__(self, other):
        return self.instance is other.instance

    def __bool__(self):
        return self.name in self.instance.attrs

    def __fspath__(self):
        return os.path.join(
            self.instance.__fspath__(),
            self.name + '.pkl'
        )

class subframe(attr):
    """
    Caches a Frame or Series as an attribute of the parent frame, preserving it across
    instances, and reindexing it to the parent frame's index.

    :param func: Function or property to call to get the subframe.
    :param pickle: Pickle the result of the function.
        Owner instance must implement __fspath__.
    :param kwargs: Parameters to pass to NDFrame.reindex
    """

    def __get__(self, instance, owner) -> NDFrame:
        res: NDFrame = super().__get__(instance, owner)
        if self not in self.aligned:
            res = res.reindex(instance.index.unique())
            self.__set__(instance, res)
            self.aligned.add(self)
        return res

    def __set_name__(self, owner, name):
        def fget(func):
            @functools.wraps(func)
            def wrapper(instance):
                if (
                        self.pickle
                        and os.path.exists(path := self.__fspath__())
                ):
                    if self.log:
                        logging.info(f'Loading {self.name} from {path}')
                    res = pd.read_pickle(path)
                else:
                    fget = func
                    if (
                            not callable(fget)
                            and hasattr(fget, '__get__')
                    ):
                        # func is some sort of property
                        fget = fget.__get__
                    res = fget(instance)
                    if self.pickle:
                        path = self.__fspath__()
                        os.makedirs(
                            os.path.dirname(path),
                            exist_ok=True
                        )
                        if self.log:
                            logging.info(f'Saving {self.name} to {path}')
                        res.to_pickle(path)
                if self.pickle:
                    res.unlink = functools.partial(os.unlink, self.__fspath__())

                return res

            return wrapper

        self.fget = fget(self.fget)
        self.aligned = WeakSet()
        super().__set_name__(owner, name)

if __name__ == '__main__':
    from pandas import DataFrame


    class Test(DataFrame):
        @attr
        @property
        def scalar(self):
            print('SCALAR')
            return 1

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
