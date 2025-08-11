from __future__ import annotations
import shutil

import dataclasses
import functools
import os
import re
from collections import deque
from functools import cached_property
from os import PathLike
from os import fspath
from pathlib import Path
from typing import Self

import pandas as pd
from toolz import curried, curry as cur, pipe

from tile2net.grid.grid.grid import Grid
from tile2net.grid.util import returns_or_assigns

if False:
    from tile2net.grid.ingrid.ingrid import InGrid


class ExtensionNotFoundError(ValueError):
    def __init__(
            self,
            path: str,
    ):
        super().__init__(f'No extension found in {path!r}')


class XYNotFoundError(ValueError):
    def __init__(
            self,
            path: str,
            missing: set[str] | list[str] | tuple[str, ...],
    ):
        missing_fmt = ', '.join(missing)
        super().__init__(
            f'indir failed to parse {path!r}; missing required characters: {missing_fmt}'
        )


def __get__(
        self: Dir,
        instance: InGrid | Dir,
        owner
) -> Dir:
    from ..ingrid import InGrid
    self.instance = instance
    if isinstance(instance, InGrid):
        self.grid = instance
    elif isinstance(instance, Dir):
        self.grid = instance.grid
    elif instance is None:
        self.grid = None
    else:
        raise TypeError(f'instance must be Grid or Dir, not {type(instance).__name__}')
    try:
        result = self.grid.__dict__[self._trace]
    except KeyError as e:
        if self.__wrapped__:
            result = self.__wrapped__(self.instance)
        else:
            try:
                path = self.instance.dir
            except (KeyError, AttributeError) as e:
                msg = (
                    f'{self.__name__!r} is not set. '
                    f'See `Grid.with_indir()` to actually set an '
                    f'input directory. See `Grid.with_source()` to download '
                    f'grid from a source into an input directory.'
                )
                raise AttributeError(msg) from e
            else:
                if isinstance(self.instance, Dir):

                    format = os.path.join(
                        self.instance.dir,
                        self.__name__,
                        self.instance.suffix.rsplit('.')[0] + '.png'
                    )

                else:
                    raise NotImplementedError

                result = self.__class__.from_format(format)

    result.__name__ = self.__name__
    result.instance = self.instance
    result.grid = self.grid
    return result


class Dir:
    instance: InGrid | Dir = None
    grid: InGrid = None
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
    def suffix(self) -> str:
        try:
            return self.__dict__['suffix']
        except KeyError:
            raise AttributeError('Indir.suffix is not set.')

    @suffix.setter
    def suffix(self, value: str | None):
        self.__dict__['suffix'] = value

    def __bool__(self):
        return self.format is not None

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
    def from_format(
            cls,
            format: str
    ) -> Self:  # noqa: A002

        # value = os.path.normpath(format)
        value = (
            Path(format)
            .expanduser()
            .resolve()
            .__str__()
        )
        indir = cls()
        indir.original = value

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

        try:
            indir.extension = value.rsplit('.', 1)[1]
        except IndexError:
            ...
            # raise ExtensionNotFoundError(value)
            # raise ValueError(f'No extension found in {value!r}')

        failed = [c for c in 'xy' if c not in CHARACTERS]
        if failed:
            raise XYNotFoundError(value, failed)
            # raise ValueError(f'indir failed to parse {value!r}')

        parts = []
        r = result[0]
        parts.extend([r[0], f'{{{r[1]}}}', r[2]])
        for r in list(result)[1:]:
            parts.extend([f'{{{r[1]}}}', r[2]])
        indir.format = ''.join(parts)
        indir.dir = parts[0].rsplit('/', 1)[0]
        indir.suffix = os.path.relpath(indir.original, indir.dir)
        return indir

    def __set__(self, instance: 'Grid', value: str | PathLike):
        if value is None:
            raise NotImplementedError
        # if isinstance(value, Path):
        #     value = str(value)
        # value = os.path.normpath(value)
        value = (
            Path(value)
            .expanduser()
            .resolve()
            .__str__()
        )

        indir = self.from_format(value)

        indir.__name__ = self.__name__
        instance.__dict__[self._trace] = indir

        indir.instance = instance
        try:
            if isinstance(instance, Grid):
                indir.grid = instance
            else:
                indir.grid = instance.grid
        except AttributeError:
            indir.grid = None

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
        except AttributeError:
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
        if (
                self.instance is None
                or isinstance(self.instance, Grid)
        ):
            return self.__name__
        elif isinstance(self.instance, Dir):
            return f'{self.instance._trace}.{self.__name__}'
        else:
            msg = f'Cannot determine trace for {self.__name__}'
            raise ValueError(msg)

    def files(
            self,
            grid: Grid,
            dirname=''
    ) -> pd.Series:
        if dirname:
            key = f'{self._trace}.{dirname}'
        else:
            key = self._trace
        if key in grid.frame:
            return grid.frame[key]
        suffix = (
            self.format
            .removeprefix(self.dir)
            .lstrip(os.sep)
        )
        format = os.path.join(self.dir, dirname, suffix)
        zoom = grid.zoom
        it = zip(grid.ytile, grid.xtile)
        data = [
            format.format(z=zoom, y=ytile, x=xtile)
            for ytile, xtile in it
        ]
        # ensure parent directories exist
        for p in {Path(p).parent for p in data}:
            p.mkdir(parents=True, exist_ok=True)
        result = pd.Series(data, index=grid.index, dtype='str')
        grid.frame[key] = result
        result = grid.frame[key]
        return result

    def _cleanup(
            self,
            base_dir: str,
            exclude: list[str]
    ):

        exclude_paths: set[str] = {
            os.path.normpath(os.path.abspath(p))
            for p in exclude
        }

        for entry in os.scandir(base_dir):
            path: str = os.path.normpath(entry.path)

            if any(
                    path == exc or path.startswith(f"{exc}{os.sep}")
                    for exc in exclude_paths
            ):
                continue

            try:
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except FileNotFoundError:
                # The target may have already been removed by another process
                continue

class TestIndir:
    indir = Dir()

    def __init__(self, indir: str | PathLike):
        self.__dict__ = {}
        self.zoom = 20
        self.extension = '.png'
        self.indir = indir



if __name__ == '__main__':
    class TestIndir:
        indir = Dir()

        def __init__(self, indir: str | PathLike):
            self.__dict__ = {}
            self.zoom = 20
            self.extension = '.png'
            self.indir = indir


    @dataclasses.dataclass
    class Tile:
        xtile: int
        ytile: int
        zoom: int


    grid = [
        Tile(1, 2, 3),
        Tile(4, 5, 6),
        Tile(7, 8, 9),
    ]
    test = TestIndir('input/dir.png')
    test = TestIndir('input/dir/x/y/z.png')
    test = TestIndir('input/dir/x_y_z.png')
    test = TestIndir('input/dir/y/x/z.png')
    test = TestIndir('input/dir/x/y.png')

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
