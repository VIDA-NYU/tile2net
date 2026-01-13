from __future__ import annotations

import os
import re
from abc import abstractmethod
from collections import deque
from functools import *
from pathlib import Path
from typing import *

import pandas as pd
import pyarrow as pa

from tile2net.grid.source.exceptions import InvalidLocalPath
from tile2net.grid.source.source import Source

if TYPE_CHECKING:
    from tile2net.grid.grid.grid import Grid


class Local(Source):

    @cached_property
    @abstractmethod
    def original(self) -> str:
        """Original path string as provided, before parsing."""

    @cached_property
    @abstractmethod
    def format(self) -> str:
        """Format string with {x}, {y}, {z} placeholders for tile coordinates."""

    @cached_property
    @abstractmethod
    def suffix(self) -> str:
        """Relative path from root directory to tile files."""

    @cached_property
    @abstractmethod
    def root(self) -> str:
        """Root directory path containing tile files."""

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension without the leading dot (e.g., 'png', 'jpg')."""

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
        template = os.path.join(self.dir, dirname, suffix)
        zoom = grid.zoom
        it = zip(grid.xtile.values, grid.ytile.values)
        obj = (
            template.format(z=zoom, y=ytile, x=xtile)
            for xtile, ytile in it
        )
        data = pa.array(obj, type=pa.string(), size=len(grid))
        for p in {Path(p).parent for p in data}:
            p.mkdir(parents=True, exist_ok=True)
        result = pd.Series(data, index=grid.index, dtype='str')
        return result

    @classmethod
    def from_str(cls, value: str) -> Self:
        instance = cls()

        path_obj = Path(value).expanduser().resolve()
        path_str = str(path_obj)
        instance.original = os.path.normpath(path_str)

        path_parts = re.split(r'[/_. \-\\]', path_str)
        single_char_parts = {
            part.casefold()
            for part in path_parts
            if len(part) == 1
        }

        characters = dict(
            x=0,
            y=1,
            z=2,
        )
        found_chars = {
            char: idx
            for char, idx in characters.items()
            if char in single_char_parts
        }

        characters = found_chars.copy()

        sep_idx = path_str.rfind(os.sep)
        dot_idx = path_str.rfind('.', sep_idx + 1)

        if dot_idx == -1:
            string = path_str
            ext = ''
        else:
            string = path_str[:dot_idx]
            ext = path_str[dot_idx + 1:]

        instance.extension = ext

        if ext:
            ext = '.' + ext

        match = cls._match(string, characters)
        instance.root = match[0]

        result = deque([match])
        while True:
            if not match[1]:
                break
            if match[1] in characters:
                del characters[match[1]]

            if not characters:
                break

            match = cls._match(match[0], characters)
            result.appendleft(match)

        failed = [
            c
            for c in 'xy'
            if c not in found_chars
        ]
        if failed:
            missing_fmt = ', '.join(failed)
            msg = f'Local path failed to parse {value!r}; missing required characters: {missing_fmt}'
            raise InvalidLocalPath(msg)

        parts = []
        r = result[0]
        parts.extend([r[0], f'{{{r[1]}}}', r[2]])
        for r in list(result)[1:]:
            parts.extend([f'{{{r[1]}}}', r[2]])

        instance.format = ''.join(parts) + ext

        instance.root = parts[0].rsplit('/', 1)[0]
        if os.sep in parts[0]:
            instance.root = parts[0].rsplit(os.sep, 1)[0]

        instance.suffix = os.path.relpath(instance.format, instance.root)

        return instance

    @classmethod
    def from_inferred(cls, value: str) -> Self:
        """Infer and construct Local instance from a path string."""
        return cls.from_str(value)

    @staticmethod
    def _match(string: str, characters: dict[str, str]):
        """Match the first occurrence of any character in 'characters' within 'string'."""
        c = '|'.join(characters)
        pattern = rf"^(.*)({c})(.*)$"
        match = re.match(pattern, string, flags=re.IGNORECASE)
        if not match:
            return string, '', ''
        return match.groups()
