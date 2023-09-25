from __future__ import annotations
import numpy as np
from numpy import ndarray

from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import copy
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import functools
import os
from _weakrefset import WeakSet
from typing import Callable, Type, Union

import numpy as np
import pandas as pd
import logging
import pickle
from pandas.core.generic import NDFrame

# todo: force fget if keyerror when setting column or subframe

__all__ = ['attr', 'subframe', 'column']


class attr:
    instance: NDFrame
    owner: Type[NDFrame]
    _validate = None

    def __init__(
            self,
            func=None,
            *,
            pickle=False,
            log=False,
            auto=False,
            step=False,
            constant=False,
            init=False,
            **kwargs
    ):
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
        self.auto = auto
        self.step = step
        self.constant = constant
        self.init = init

    def set(self, instance, value):
        instance.attrs[self.name] = value

    def __set__(self, instance: NDFrame, value):
        self.instance = instance
        self.owner = type(instance)
        if self._validate is not None:
            value = self._validate(instance, value)
        self.set(instance, value)

    @classmethod
    def validate(cls, func):
        self = cls()
        self._validate = func
        return self

    def get(self, instance, owner):
        return instance.attrs[self.name]

    def __get__(self, instance: NDFrame, owner):
        self.instance = instance
        self.owner = owner
        if instance is None:
            return self
        if not self:
            res = self.fget(instance)
            self.__set__(instance, res)
        return self.get(instance, owner)

    def delete(self, instance):
        if self.constant:
            raise UserWarning(
                f'Deleting constant attribute {self.name}'
            )
        try:
            del instance.attrs[self.name]
        except KeyError:
            pass

    def __delete__(self, instance: NDFrame):
        self.instance = instance
        self.owner = type(instance)
        self.delete(instance)

    def __call__(self, obj):
        if isinstance(obj, NDFrame):
            # attr is being used as unbound method
            return self.__get__(obj, type(obj))
        # attr is wrapping a method
        if (
                not callable(obj)
                and hasattr(obj, '__get__')
        ):
            # obj is some sort of property
            obj = obj.__get__
        self.fget = obj
        return self

    def __set_name__(self, owner, name):
        self.name = name

        def fget(func):
            @functools.wraps(func)
            def wrapper(instance):
                # todo: we need to be able to redo fget if index difference
                if (
                        self.pickle
                        and os.path.exists(path := self.__fspath__())
                ):
                    raise NotImplementedError
                    if self.log:
                        logging.info(f'Loading {self.name} from {path}')
                    with open(path, 'rb') as f:
                        res = pickle.load(f)
                else:
                    if self.log:
                        logging.info(f'Calculating {self.name}')
                    fget = func
                    if (
                            not callable(fget)
                            and hasattr(fget, '__get__')
                    ):
                        # func is some sort of property
                        fget = fget.__get__
                    res = fget(instance)
                    if self.step:
                        object.__setattr__(res, self.name, res)
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

        if self.auto:
            @functools.wraps(owner.__init__)
            def init(*args, **kwargs):
                owner.__init__(*args, **kwargs)
                self.__get__(*args, **kwargs)

            owner.__init__ = init

    # def __new__(cls, *args, **kwargs):
    #     def __get__(func: Callable) -> Callable:
    #         # @functools.wraps(func)
    #         def wrapper(self: attr, instance, owner):
    #             # saves if not on disk, loads if on disk
    #             self.instance = instance
    #             self.owner = type(instance)
    #             if instance is None:
    #                 return self
    #
    #             if not self:
    #                 res = self.fget(instance)
    #                 self.__set__(instance, res)
    #
    #             res = func(self, instance, owner)
    #             return res
    #
    #         return wrapper
    #
    #     def __set__(func: Callable) -> Callable:
    #         # @functools.wraps(func)
    #         def wrapper(self: attr, instance: NDFrame, value):
    #             self.instance = instance
    #             self.owner = type(instance)
    #             func(self, instance, value)
    #
    #         return wrapper
    #
    #     def __delete__(func: Callable) -> Callable:
    #         # @functools.wraps(func)
    #         def wrapper(self: attr, instance: NDFrame):
    #             self.instance = instance
    #             self.owner = type(instance)
    #             func(self, instance)
    #
    #         return wrapper
    #
    #     # cls.__get__ = __get__(cls.__get__)
    #     # cls.__set__ = __set__(cls.__set__)
    #     # cls.__delete__ = __delete__(cls.__delete__)
    #     self = super().__new__(cls)
    #     self.__get__ = __get__(self.__get__)
    #     self.__set__ = __set__(self.__set__)
    #     self.__delete__ = __delete__(self.__delete__)
    #     return self
    #

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

    # def __get__(self, instance, owner) -> NDFrame:
    #     res: NDFrame = super().__get__(instance, owner)
    #     if self not in self.aligned:
    #         # res = res.reindex(instance.index.unique(), )
    #         loc = instance.index.unique().intersection(res.index)
    #         res = res.loc[loc]
    #         self.__set__(instance, res)
    #         self.aligned.add(self)
    #     return res

    # def __get__(self, instance: DataFrame, owner) -> Union[subframe, NDFrame]:
    #     unaligned = super().__get__(instance, owner)
    #     if unaligned is self:
    #         return self
    #     if self not in self.aligned:
    #         # loc = instance.index.unique().intersection()
    #         loc = (
    #             instance.index
    #             .unique()
    #             .intersection(unaligned.index)
    #         )
    #         aligned = unaligned.loc[loc]
    #         self.__set__(instance, aligned)
    #         # self.aligned.add(self)

    def get(self, instance, owner):
        unaligned = super().get(instance, owner)
        if unaligned is self:
            return self
        if self.name not in instance.__dict__:
            loc = (
                instance.index
                .unique()
                .intersection(unaligned.index)
            )
            aligned = unaligned.loc[loc]
            instance.__dict__[self.name] = aligned
        aligned = instance.__dict__[self.name]
        return aligned

    def delete(self, instance):
        super().delete(instance)
        if self.name in instance.__dict__:
            del instance.__dict__[self.name]

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
        super().__set_name__(owner, name)


class column(attr):
    def set(self, instance, value):
        value.__class__.mro()
        # Series.mro()
        # ndarray.mro()
        # if isinstance(value, ndarray):
        #     assert len(value) == len(instance)
        # elif not (
        #         value.index
        #                 .difference(instance.index)
        #                 .empty
        # ):
        #     # todo: perhaps force it to recompute
        #     raise ValueError('Cannot assign a Series with a different index')
        if (
            isinstance(value, Series)
            and not value.index.difference(instance.index).empty
        ):
            raise ValueError('Cannot assign a Series with a different index')


        instance[self.name] = value

    def delete(self, instance):
        try:
            del instance[self.name]
        except KeyError:
            ...


    def get(self, instance: DataFrame, owner):
        # return instance[self.name]
        if self.name in instance.index.names:
            return instance.index.get_level_values(self.name)
        return instance[self.name]

    def __bool__(self):
        # return self.name in self.instance.columns
        # noinspection PyTypeChecker
        instance: DataFrame = self.instance
        if self.name in instance.index.names:
            return True
        return self.name in instance.columns



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

        @column
        @property
        def col(self):
            print('COL')
            return np.arange(len(self))

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
