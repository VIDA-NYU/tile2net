from __future__ import annotations

import functools
from typing import Optional, Self
import prometheus_client
from typing import Self
from functools import cached_property

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from os import fspath

import dataclasses
import imageio.v3 as iio
import numpy as np
import os
import pandas as pd
import re
from numpy import ndarray
from pathlib import Path
from toolz import curried, curry as cur, pipe
from typing import Iterator
from typing import Union
from .util import returns_or_assigns

if False:
    from .tiles import Tiles


def __get__(
        self: Dir,
        instance: Union[
            Tiles, Dir
        ],
        owner
):
    self.instance = instance
    self.owner = owner
    from .tiles import Tiles
    if isinstance(instance, Tiles):
        self.tiles = instance
    elif isinstance(instance, Dir):
        self.tiles = instance.tiles
    elif instance is None:
        self.tiles = None
    else:
        raise TypeError(f'instance must be Tiles or Dir, not {type(instance).__name__}')

    try:
        result = self.tiles.attrs[self._trace]
    except KeyError as e:
        try:
            dir = self.instance.dir
        except (KeyError, AttributeError) as e:
            msg = (
                f'{self.__name__!r} is not set. '
                f'See `Tiles.with_indir()` to actually set an '
                f'input directory. See `Tiles.with_source()` to download '
                f'tiles from a source into an input directory.'
            )
            raise AttributeError(msg) from e
        else:
            if self.__wrapped__:
                result = self.__wrapped__(self.instance)
            else:
                path = os.path.join(dir, self.__name__)
                result = self.__class__.from_dir(path)
            self.tiles.attrs[self._trace] = result
    result.__name__ = self.__name__
    result.instance = self.instance
    result.owner = self.owner
    result.tiles = self.tiles
    return result


class Dir:
    tiles: Optional[Tiles] = None
    instance: Tiles | Dir
    owner: type[Tiles] | type[Dir]
    locals().update(
        __get__=__get__
    )
    __wrapped__ = None

    def __set_name__(self, owner, name):
        self.__name__ = name

    characters = {
        c: i
        for i, c in enumerate('xyz')
    }

    @property
    def format(self) -> str | None:
        try:
            return self.__dict__['format']
        except KeyError:
            raise AttributeError('Indir.format is not set.')

    @format.setter
    def format(self, value: str | None):
        self.__dict__['format'] = value

    @property
    def root(self) -> str | None:
        # todo: why did I implement this?
        try:
            return self.__dict__['root']
        except KeyError:
            raise AttributeError('Indir.root is not set.')

    @root.setter
    def root(self, value: str | None):
        self.__dict__['root'] = value

    @property
    def original(self) -> str | None:
        try:
            return self.__dict__['original']
        except KeyError:
            raise AttributeError('Indir.original is not set.')

    @original.setter
    def original(self, value: str | Path | PathLike | None):
        if isinstance(value, (Path, PathLike)):
            value = fspath(value)
        self.__dict__['original'] = os.path.normpath(value) if value is not None else value

    @property
    def extension(self) -> str | None:
        try:
            return self.__dict__['extension']
        except KeyError:
            raise AttributeError('Indir.extension is not set.')

    @extension.setter
    def extension(self, value: str | None):
        self.__dict__['extension'] = value

    @property
    def suffix(self):
        try:
            return self.__dict__['suffix']
        except KeyError:
            raise AttributeError('Indir.suffix is not set.')

    @suffix.setter
    def suffix(self, value: str | None):
        self.__dict__['suffix'] = value

    def __bool__(self):
        return self.format is not None

    # def __set__(
    #         self,
    #         instance: Tiles,
    #         value: str | PathLike
    # ):
    #
    #     if value is None:
    #         raise NotImplementedError
    #     if isinstance(value, Dir):
    #         # result = type(value)(value)
    #         # result.__name__ = self.__name__
    #         # instance.attrs[self._trace] = result
    #         raise NotImplementedError
    #         return
    #     if isinstance(value, Path):
    #         value = str(value)
    #     value = os.path.normpath(value)
    #     indir = self.__class__()
    #     instance.attrs[self._trace] = indir
    #
    #     indir.original = value
    #     indir.__name__ = self.__name__
    #
    #     try:
    #         indir.extension = value.rsplit('.', 1)[1]
    #     except IndexError:
    #         raise ValueError(f'No extension found in {value!r}')
    #
    #     CHARACTERS: dict[str, str] = pipe(
    #         re.split(r'[/_. \-\\]', value),
    #         curried.filter(lambda c: len(c) == 1),
    #         curried.map(str.casefold),
    #         set,
    #         cur(set.__contains__),
    #         curried.keyfilter(d=indir.characters),
    #     )
    #     characters = CHARACTERS.copy()
    #
    #     string = value
    #     match = indir._match(string, characters)
    #     indir.root = match[0]
    #     result = deque([match])
    #
    #     while True:
    #         c = match[1]
    #         if not c:
    #             break
    #         del characters[match[1]]
    #         if not characters:
    #             break
    #         match = indir._match(match[0], characters)
    #         result.appendleft(match)
    #
    #     failed = [
    #         c
    #         for c in 'xy'
    #         if c not in CHARACTERS
    #     ]
    #     if failed:
    #         raise ValueError(
    #             f'indir failed to parse {value!r} '
    #         )
    #
    #     parts = []
    #     r = result[0]
    #     parts.append(r[0])
    #     parts.append(f'{{{r[1]}}}')
    #     parts.append(r[2])
    #     for r in list(result)[1:]:
    #         parts.append(f'{{{r[1]}}}')
    #         parts.append(r[2])
    #     format = ''.join(parts)
    #     indir.format = format
    #     indir.dir = parts[0].rsplit('/', 1)[0]
    #     indir.suffix = os.path.relpath(indir.original, indir.dir)
    #
    #     from .tiles import Tiles
    #     indir.instance = instance
    #     owner = type(instance)
    #     indir.owner = owner
    #     if issubclass(owner, Tiles):
    #         indir.tiles = instance
    #         indir.Tiles = owner
    #     else:
    #         indir.tiles = instance.tiles
    #         indir.Tiles = instance.Tiles

    @classmethod
    def from_dir(cls, dir: str) -> Self:  # noqa: N803
        indir = cls()
        dir = os.path.normpath(dir)
        indir.original = dir
        indir.root = dir
        indir.dir = dir
        indir.format = None
        indir.extension = None
        indir.suffix = ''
        return indir

    @classmethod
    def from_format(cls, format: str) -> Self:  # noqa: A002
        value = os.path.normpath(format)
        indir = cls()
        indir.original = value
        try:
            indir.extension = value.rsplit('.', 1)[1]
        except IndexError:
            raise ValueError(f'No extension found in {value!r}')

        CHARACTERS: dict[str, str] = pipe(
            re.split(r'[/_. \-\\]', value),
            curried.filter(lambda c: len(c) == 1),
            curried.map(str.casefold),
            set,
            cur(set.__contains__),
            curried.keyfilter(d=indir.characters),
        )
        characters = CHARACTERS.copy()

        string = value
        match = indir._match(string, characters)
        indir.root = match[0]
        result = deque([match])

        while True:
            if not match[1]:
                break
            del characters[match[1]]
            if not characters:
                break
            match = indir._match(match[0], characters)
            result.appendleft(match)

        failed = [c for c in 'xy' if c not in CHARACTERS]
        if failed:
            raise ValueError(f'indir failed to parse {value!r}')

        parts = []
        r = result[0]
        parts.extend([r[0], f'{{{r[1]}}}', r[2]])
        for r in list(result)[1:]:
            parts.extend([f'{{{r[1]}}}', r[2]])
        indir.format = ''.join(parts)
        indir.dir = parts[0].rsplit('/', 1)[0]
        indir.suffix = os.path.relpath(indir.original, indir.dir)
        return indir

    def __set__(self, instance: 'Tiles', value: str | PathLike):
        if value is None:
            raise NotImplementedError
        if isinstance(value, Path):
            value = str(value)
        value = os.path.normpath(value)

        indir = self.from_format(value)

        indir.__name__ = self.__name__
        instance.attrs[self._trace] = indir

        from .tiles import Tiles  # local import to avoid cyclic dependency
        indir.instance = instance
        owner = type(instance)
        indir.owner = owner
        if issubclass(owner, Tiles):
            indir.tiles = instance
        else:
            indir.tiles = instance.tiles

    @cached_property
    def dir(self) -> str:
        ...

    @staticmethod
    def _match(string: str, characters: dict[str, str]):
        c = '|'.join(characters)
        pattern = rf"^(.*)({c})(.*)$"
        match = re.match(pattern, string)
        return match.groups()

    def __repr__(self):
        try:
            return (
                f'{self.__class__.__name__}(\n'
                f'    format={self.format!r},\n'
                # f'    root={self.root!r},\n'
                f'    extension={self.extension!r}\n'
                f'    dir={self.dir!r}\n'
                f'    original={self.original!r},\n'
                f')'
            )
        except (AttributeError):
            return (
                f'{self.__class__.__name__}(\n'
                f'    dir={self.dir!r}\n'
                f'    original={self.original!r},\n'
                f')'
            )

    def __init__(self, func=None):
        if returns_or_assigns(func):
            functools.update_wrapper(self, func)

    @classmethod
    def is_valid(cls, indir: str):
        try:
            TestIndir(indir)
        except ValueError:
            return False
        else:
            return True

    @cached_property
    def _trace(self):
        from .tiles import Tiles
        if (
                self.instance is None
                or isinstance(self.instance, Tiles)
        ):
            return self.__name__
        elif isinstance(self.instance, Dir):
            return f'{self.instance._trace}.{self.__name__}'
        else:
            msg = (
                f'Cannot determine trace for {self.__name__} in '
                f'{self.owner} with {self.instance=}'
            )
            raise ValueError(msg)

    @property
    def files(self) -> pd.Series:
        tiles = self.tiles.stitched
        key = self._trace
        if key in tiles:
            return tiles[key]
        else:
            format = self.format
            zoom = tiles.zoom
            it = zip(tiles.ytile, tiles.xtile)
            data = [
                format.format(z=zoom, y=ytile, x=xtile)
                for ytile, xtile in it
            ]
            result = pd.Series(data, index=tiles.index)
            tiles[key] = result
            return tiles[key]


class TestIndir:
    indir = Dir()

    def __init__(self, indir: str | PathLike):
        self.attrs = {}
        self.zoom = 20
        self.extension = '.png'
        self.indir = indir


class Loader:
    def __init__(
            self,
            files: pd.Series,
            group: pd.Series,
            row: pd.Series,
            col: pd.Series,
            tile: tuple[int, int, int],
            mosaic: tuple[int, int, int],
    ):
        if not (len(files) == len(group) == len(row) == len(col)):
            raise ValueError('files, group, row, and col must be the same length')

        self.files = files.reset_index(drop=True)
        self.group = group.reset_index(drop=True)
        self.row = row.reset_index(drop=True)
        self.col = col.reset_index(drop=True)

        self.tile_h, self.tile_w, self.tile_c = tile
        self.mos_h, self.mos_w, self.mos_c = mosaic

        if self.mos_h % self.tile_h or self.mos_w % self.tile_w:
            raise ValueError('mosaic dimensions are not multiples of tile size')

        # dtype discovery from first existing file
        try:
            sample_path = next(p for p in self.files if Path(p).is_file())
            self.dtype = iio.imread(sample_path).dtype
        except StopIteration:
            self.dtype = np.uint8  # default

        # sort by group, then row, then col for deterministic iteration
        order = np.lexsort((self.col, self.row, self.group))
        self.files = self.files.iloc[order]
        self.group = self.group.iloc[order]
        self.row = self.row.iloc[order]
        self.col = self.col.iloc[order]

        self.unique_groups = self.group.unique()
        self.ncols = self.mos_w // self.tile_w
        self.nrows = self.mos_h // self.tile_h

    @staticmethod
    def _read(path: str | os.PathLike) -> ndarray | None:
        try:
            if Path(path).is_file():
                return iio.imread(path)
        except Exception:
            pass
        return None

    def __iter__(self) -> Iterator[ndarray]:

        with ThreadPoolExecutor() as pool:
            for g in self.unique_groups:
                mask = self.group == g
                f_group = self.files[mask]
                r_group = self.row[mask].to_numpy()
                c_group = self.col[mask].to_numpy()

                out = np.zeros((self.mos_h, self.mos_w, self.mos_c), dtype=self.dtype)

                fut2idx = {
                    pool.submit(self._read, p): idx
                    for idx, p in enumerate(f_group)
                }

                for fut in as_completed(fut2idx):
                    img = fut.result()
                    if img is None:
                        continue
                    idx = fut2idx[fut]
                    r = r_group[idx]
                    c = c_group[idx]

                    y0 = r * self.tile_h
                    x0 = c * self.tile_w
                    out[y0:y0 + self.tile_h, x0:x0 + self.tile_w, :img.shape[2] if img.ndim == 3 else ...] = img

                yield out


if __name__ == '__main__':
    class TestIndir:
        indir = Dir()

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

Dir.__get__
