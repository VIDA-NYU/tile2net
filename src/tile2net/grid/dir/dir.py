from __future__ import annotations

import copy
import os
import re
from abc import abstractmethod
from collections import deque
from functools import *
from pathlib import Path
from typing import *

import pandas as pd

from tile2net.grid.dir.exceptions import XYNotFoundError
from tile2net.grid.frame.namespace import namespace
from tile2net.grid.frame.weak import weak

if TYPE_CHECKING:
    from ..grid.grid import Grid
    from ..ingrid.ingrid import InGrid


class Dir(
    namespace
):
    """
    Class responsible for managing directory-related operations and attributes.

    This class provides functionality to interact with directories, including
    format processing, extensions, suffix management, and other properties.
    It is designed to handle complex directory structures and offer utilities
    to represent or manipulate paths effectively.
    """

    @cached_property
    @abstractmethod
    def format(self) -> str:
        ...

    @cached_property
    @abstractmethod
    def root(self) -> str:
        ...

    @cached_property
    @abstractmethod
    def original(self):
        ...

    @cached_property
    @abstractmethod
    def extension(self) -> str:
        ...

    @cached_property
    @abstractmethod
    def suffix(self) -> str:
        ...

    @cached_property
    @abstractmethod
    def dir(self) -> str:
        ...

    @weak.property
    @abstractmethod
    def ingrid(self) -> InGrid:
        ...

    @overload
    def __get__[T](
            self,
            instance,
            owner: type[T],
    ) -> T:
        ...

    @overload
    def __get__[T](
            self,
            instance: T,
            owner,
    ) -> T:
        ...

    def __get__(
            self,
            instance: Dir,
            owner
    ) -> Self:
        if instance is None:
            out = self
        elif not isinstance(instance, Dir):
            raise TypeError(instance)
        else:
            cache = instance.__dict__
            name = self.__name__
            if name not in cache:
                if self.__wrapped__:
                    value = self.__wrapped__(instance)
                else:
                    value = self.from_parent(instance, name, self.extension)
                self.__set__(instance, value=value)
            out = cache[name]
        return out

    def __set__(self, instance: Dir, value) -> Self:
        if isinstance(value, str):
            value = self.from_format(value)
        if not isinstance(value, Dir):
            raise TypeError(value)
        instance.__dict__[self.__name__] = copy.copy(value)

    def __delete__(self, instance):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            ...

    def __bool__(self):
        return self.format is not None

    def __repr__(self):
        attrs = []
        for attr in ['format', 'extension', 'dir', 'original', 'suffix']:
            try:
                value = getattr(self, attr)
                if value is not None:
                    attrs.append(f'    {attr}={value!r}')
            except AttributeError:
                continue
        return f'{self.__class__.__name__}(\n' + ',\n'.join(attrs) + '\n)'

    @classmethod
    def from_parent(
            cls,
            parent: Dir,
            name: str,
            extension: str = None,
    ) -> Self:
        if not extension:
            extension = parent.extension
        suffix = (
                parent.suffix
                .rsplit('.')
                [0]
                + f'.{extension}'
        )
        item = os.path.join(parent.dir, name, suffix)
        out = cls.from_format(item)
        return out

    @classmethod
    def from_format(
            cls,
            item: str,
            force_xy: bool = True,
    ) -> Self:
        # todo: clean this up. This should be more clear.

        value = (
            Path(item)
            .expanduser()
            .resolve()
            .__str__()
        )
        indir = cls()
        indir.original = value

        keys_found = {
            token.casefold()
            for token in re.split(r'[/_. \-\\]', value)
            if len(token) == 1
        }

        CHARACTERS = {
            k: v
            for k, v in indir.characters.items()
            if k in keys_found
        }
        characters = CHARACTERS.copy()

        # string, ext = value.rsplit('.', 1)
        sep_idx = value.rfind(os.sep)
        # find last dot *after* last separator
        dot_idx = value.rfind('.', sep_idx + 1)

        if dot_idx == -1:
            # no dot after last separator
            string = value
            ext = ''
        else:
            string = value[:dot_idx]
            ext = value[dot_idx + 1:]
        if ext:
            ext = '.' + ext
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

        failed = [
            c
            for c in 'xy'
            if c not in CHARACTERS
        ]
        if force_xy and failed:
            raise XYNotFoundError(value, failed)

        parts = []
        r = result[0]
        parts.extend([r[0], f'{{{r[1]}}}', r[2]])
        for r in list(result)[1:]:
            parts.extend([f'{{{r[1]}}}', r[2]])
        indir.format = ''.join(parts) + (ext if ext else '')
        indir.dir = parts[0].rsplit('/', 1)[0]
        indir.suffix = os.path.relpath(indir.original, indir.dir)
        return indir

    def with_suffix(
            self,
            suffix: str = '/z/x_y'
    ):

        format = os.path.join(
            self.dir,
            suffix,
        )
        result = self.__class__.from_format(format)
        return result

    def files(
            self,
            grid: Grid,
            dirname=''
    ) -> pd.Series:
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
        return result
