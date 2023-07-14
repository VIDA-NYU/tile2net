from __future__ import annotations

import dataclasses
import re
from collections import deque
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Type

from numpy import ndarray
from toolz import curried, curry as cur, pipe

from tile2net.raster.util import cached_descriptor


if False:
    from tile2net.raster.raster import Raster
    from tile2net.raster.tile import Tile

class InputDir:
    @cached_descriptor
    def format(self):
        ...

    @cached_descriptor
    def root(self):
        ...

    @cached_descriptor
    def original(self):
        ...

    @cached_descriptor
    def extension(self):
        ...

    def _match(self, string: str, characters: dict[str]):
        c = '|'.join(characters)
        pattern = rf"^(.*)({c})(.*)$"
        match = re.match(pattern, string)
        return match.groups()

    characters = {
        c: i
        for i, c in enumerate('xyz')
    }

    def __set__(self, instance: Raster, value: str | PathLike):
        if value is None:
            return
        if isinstance(value, Path):
            value = str(value)

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

        # if not (
        #     'x' in CHARACTERS
        #     and 'y' in CHARACTERS
        # ):
        #     raise ValueError(f'{self.name} failed to parse ')
        failed = [
            c
            for c in 'xy'
            if c not in CHARACTERS
        ]
        if failed:
            raise ValueError(
                f'{self.name} failed to parse {value!r} '
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

    # noinspection PyMethodOverriding
    def __get__(self, instance: Raster, owner: Type[Raster]) -> InputDir:
        from tile2net.raster.raster import Raster
        self.raster: Raster = instance
        self.Raster: Type[Raster] = owner
        return self

    def __delete__(self, instance):
        del self.format

    def __call__(self, tiles: ndarray = None) -> Iterator[Path]:
        if tiles is None:
            tiles = self.raster.tiles
        if isinstance(tiles, ndarray):
            tiles: Iterable[Tile] = tiles.flat
        format = self.format
        for tile in tiles:
            res = format.format(
                x=tile.xtile,
                y=tile.ytile,
                z=tile.zoom,
            )
            yield Path(res)

    def __bool__(self):
        return self.format is not None

    def __fspath__(self):
        return self.dir

    def __repr__(self):
        return self.original

    def __str__(self):
        return self.original

    def __set_name__(self, owner, name):
        self.name = name

if __name__ == '__main__':
    class Test:
        input_dir = InputDir()

        def __init__(self, input_dir: str | PathLike):
            self.zoom = 20
            self.extension = '.png'
            self.input_dir = input_dir

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

    list(test.input_dir(tiles))
    test = Test('input/dir/x/y/z.png')
    test = Test('input/dir/x_y_z.png')
    test = Test('input/dir/y/x/z.png')
    test = Test('input/dir/x/y.png')

    try:
        test = Test('input/dir/x.png')
    except ValueError:
        pass
    else:
        raise AssertionError
    try:
        test = Test('input/dir/x.png')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError
    try:
        test = Test('input/dir/xy.png')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError
    try:
        test = Test('input/dir/x_y_z')
    except ValueError as e:
        print(e)
    else:
        raise AssertionError

    test = Test('input/dir/arst_x_y_z.png')
    test