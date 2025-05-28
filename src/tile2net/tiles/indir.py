from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import imageio.v3 as iio
import numpy as np
import pandas as pd
from numpy import ndarray
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import dataclasses
import re
from collections import deque
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Type

import pandas as pd
from numpy import ndarray
from toolz import curried, curry as cur, pipe

from tile2net.raster.util import cached_descriptor
import os

if False:
    from .tiles import Tiles
    from tile2net.raster.raster import Raster


class Indir:
    tiles: Tiles

    characters = {
        c: i
        for i, c in enumerate('xyz')
    }

    # def files(
    #         self,
    #         tiles: Tiles
    # ) -> pd.Series[str]:
    @property
    def files(self) -> pd.Series:
        tiles = self.tiles
        format = self.format
        zoom = tiles.zoom
        it = zip(tiles.ytile, tiles.xtile)
        data = [
            format.format(z=zoom, y=ytile, x=xtile)
            for ytile, xtile in it
        ]
        result = pd.Series(data, index=tiles.index, name='path')
        return result

    @property
    def format(self):
        return self.tiles.attrs.get('indir.format')

    @format.setter
    def format(self, value: str):
        self.tiles.attrs['indir.format'] = value

    @property
    def root(self):
        return self.tiles.attrs.get('indir.root')

    @root.setter
    def root(self, value: str):
        self.tiles.attrs['indir.root'] = str(value)

    @property
    def original(self):
        return self.tiles.attrs.get('indir.original')

    @original.setter
    def original(self, value: str | PathLike):
        if isinstance(value, Path):
            value = str(value)
        self.tiles.attrs['indir.original'] = os.path.normpath(value)

    @property
    def extension(self):
        return self.tiles.attrs.get('indir.extension')

    @extension.setter
    def extension(self, value: str):
        self.tiles.attrs['indir.extension'] = value

    def __get__(
            self,
            instance,
            owner
    ):
        self.tiles = instance
        self.Tiles = owner
        return self

    def __bool__(self):
        return self.format is not None

    def __set__(
            self,
            instance: Raster,
            value: str | PathLike
    ):
        self.tiles = instance
        self.Tiles = type(instance)
        if value is None:
            return
        if isinstance(value, Path):
            value = str(value)
        value = os.path.normpath(value)

        self.original = value

        try:
            self.extension = value.rsplit('.', 1)[1]
        except IndexError:
            raise ValueError(f'No extension found in {value!r}')

        CHARACTERS: dict[str] = pipe(
            re.split(r'[/_. \-\\]', value),
            curried.filter(lambda c: len(c) == 1),
            curried.map(str.casefold),
            set,
            cur(set.__contains__),
            curried.keyfilter(d=self.characters),
        )
        characters = CHARACTERS.copy()

        string = value
        match = self._match(string, characters)
        self.root = match[0]
        result = deque([match])

        while True:
            c = match[1]
            if not c:
                break
            del characters[match[1]]
            if not characters:
                break
            match = self._match(match[0], characters)
            result.appendleft(match)

        failed = [
            c
            for c in 'xy'
            if c not in CHARACTERS
        ]
        if failed:
            raise ValueError(
                f'indir failed to parse {value!r} '
            )

        parts = []
        r = result[0]
        parts.append(r[0])
        parts.append(f'{{{r[1]}}}')
        parts.append(r[2])
        for r in list(result)[1:]:
            parts.append(f'{{{r[1]}}}')
            parts.append(r[2])
        format = ''.join(parts)
        self.format = format
        self.dir = parts[0].rsplit('/', 1)[0]

    @staticmethod
    def _match(string: str, characters: dict[str, str]):
        c = '|'.join(characters)
        pattern = rf"^(.*)({c})(.*)$"
        match = re.match(pattern, string)
        return match.groups()

    def __init__(self, *args, **kwargs):
        ...

    @classmethod
    def is_valid(cls, indir: str):
        try:
            TestIndir(indir)
        except ValueError:
            return False
        else:
            return True


class TestIndir:
    indir = Indir()

    def __init__(self, indir: str | PathLike):
        self.attrs = {}
        self.zoom = 20
        self.extension = '.png'
        self.indir = indir


class Loader:
    def __init__(
            self,
            files: pd.Series,
            fallback: ndarray
    ):
        msg = f'All file groups must have the same length.'
        assert (
                files
                .groupby(level=files.index.names)
                .size()
                .nunique()
                == 1
        ), msg
        self.files = files
        self.fallback = fallback

    @staticmethod
    def _read(path: str | os.PathLike, fallback: ndarray) -> ndarray:
        try:
            return iio.imread(path) if Path(path).is_file() else fallback
        except Exception:
            return fallback

    def __iter__(self) -> Iterator[ndarray]:
        executor = ThreadPoolExecutor()
        files = self.files
        groups = (
            files
            .groupby(level=files.index.names, sort=False)
            .__iter__()
        )

        try:
            _, current_series = next(groups)
        except StopIteration:
            executor.shutdown(wait=True)
            return

        current_futures = [
            executor.submit(self._read, p, self.fallback)
            for p in current_series
        ]

        for _, next_series in groups:
            next_futures = [
                executor.submit(self._read, p, self.fallback)
                for p in next_series
            ]
            arrays = [
                f.result()
                for f in current_futures
            ]
            yield np.stack(arrays, axis=0)
            current_futures = next_futures

        yield np.stack([
            f.result()
            for f in current_futures
        ], axis=0)
        executor.shutdown(wait=True)


if __name__ == '__main__':
    class TestIndir:
        indir = Indir()

        def __init__(self, indir: str | PathLike):
            self.attrs = {}
            self.zoom = 20
            self.extension = '.png'
            self.indir = indir


    @dataclasses.dataclass
    class Tile:
        xtile: int
        ytile: int
        zoom: int


    tiles = [
        Tile(1, 2, 3),
        Tile(4, 5, 6),
        Tile(7, 8, 9),
    ]
    test = TestIndir('input/dir/x/y/z.png')
    test = TestIndir('input/dir/x_y_z.png')
    test = TestIndir('input/dir/y/x/z.png')
    test = TestIndir('input/dir/x/y.png')
    # test = TestIndir('input/dir/x/')

    try:
        test = TestIndir('input/dir/x.png')
    except ValueError:
        pass
    else:
        raise AssertionError
    try:
        test = TestIndir('input/dir/x.png')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError
    try:
        test = TestIndir('input/dir/xy.png')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError
    try:
        test = TestIndir('input/dir/x_y_z')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError

    test = TestIndir('input/dir/arst_x_y_z.png')
    test
