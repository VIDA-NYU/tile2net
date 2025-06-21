from __future__ import annotations

import functools
from typing import *

import pandas as pd
from tile2net.tiles.tiles.tiles import  Tiles

if False:
    from .dir import Dir


def __get__(
        self: BatchIterator,
        instance: Dir | Tiles,
        owner
) -> BatchIterator:
    if instance is None:
        return self
    if isinstance(instance, Dir):
        self.tiles = instance.tiles
        self.trace = instance._trace
    elif isinstance(instance, Tiles):
        self.tiles = instance
        self.trace = self.__name__
    else:
        raise TypeError(instance)
    self.instance = instance
    return self


class BatchIterator:
    __wrapped__: Callable
    tiles: Tiles = None
    key: Any = None
    trace: str = None
    instance: Dir | Tiles = None
    locals().update(
        __get__=__get__,
    )

    def __init__(
            self,
            func=None,
            tiles: Tiles = None,
            key: Any = None,
            series: pd.Series = None,
    ):
        if func is None:
            ...
        elif callable(func):
            functools.update_wrapper(self, func)
            return
        else:
            raise TypeError()
        if tiles is not None:
            self.tiles = tiles
        if key is not None:
            self.key = key

        cfg = tiles.cfg
        n = cfg.model.bs_val
        if not cfg.force:
            loc = ~tiles.skip
            series = series.loc[loc]
        a = series.values
        q, r = divmod(len(a), n)

        def gen():
            for i in range(0, q * n, n):
                yield a[i:i + n]
            if r:
                yield a[-r:]

        cache = tiles.__dict__['iterator']
        self.gen = gen()
        self.cache = cache
        self.key = key

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __delete__(
            self,
            instance: Dir,
    ):
        try:
            del instance.tiles.__dict__['iterator']
        except KeyError:
            pass

    def __call__(
            self,
            *args,
    ):
        key = self.trace, *args
        tiles = self.tiles
        cache = tiles.__dict__.setdefault('iterator', {})
        if key in cache:
            return cache[key]

        series = self.__wrapped__(self.instance)
        if not isinstance(series, pd.Series):
            raise TypeError(
                f'BatchIterator must return a pandas Series, '
                f'got {type(series)}'
            )

        result = BatchIterator(
            tiles=tiles,
            key=key,
            series=series,
        )
        cache[key] = result
        return result

    def __iter__(self):
        return self

    def __next__(self) -> pd.Series:
        try:
            return next(self.gen)
        except StopIteration:
            self.cache.pop(self.key, None)
            raise
