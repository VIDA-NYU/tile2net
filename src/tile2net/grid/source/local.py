from __future__ import annotations

import os
import re
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import Self

import pandas as pd
from toolz import curried, curry as cur, pipe

from tile2net.grid.source.source import Source


class Local(Source):
    characters = {
        c: i
        for i, c in enumerate('xyz')
    }

    @cached_property
    def original(self) -> str:
        raise AttributeError(
            f'{self.__class__.__name__}.original is not set. '
            f'This attribute must be initialized via from_str().'
        )

    @cached_property
    def format(self) -> str:
        raise AttributeError(
            f'{self.__class__.__name__}.format is not set. '
            f'This attribute must be initialized via from_str().'
        )

    @cached_property
    def suffix(self) -> str:
        raise AttributeError(
            f'{self.__class__.__name__}.suffix is not set. '
            f'This attribute must be initialized via from_str().'
        )

    @cached_property
    def root(self) -> str:
        raise AttributeError(
            f'{self.__class__.__name__}.root is not set. '
            f'This attribute must be initialized via from_str().'
        )

    @cached_property
    def extension(self) -> str:
        raise AttributeError(
            f'{self.__class__.__name__}.extension is not set. '
            f'This attribute must be initialized via from_str().'
        )

    @property
    def files(self) -> pd.Series:
        grid = self.grid
        if grid is None:
            raise ValueError("Local source is not attached to a grid.")

        suffix = (
            self.format
            .removeprefix(self.root)
            .lstrip(os.sep)
        )
        fmt = os.path.join(self.root, suffix)
        zoom = grid.zoom

        data = [
            fmt.format(z=zoom, y=tile.ytile, x=tile.xtile)
            for tile in grid.tiles
        ]

        return pd.Series(data, index=grid.index, dtype='str')

    @classmethod
    def accepts(cls, value: str) -> bool:
        if value.startswith(('http://', 'https://', 's3://', 'gs://')):
            return False
        try:
            cls.from_str(value)
        except (ValueError, OSError):
            return False
        return True

    @classmethod
    def from_str(cls, value: str) -> Self:
        instance = cls()

        path_obj = Path(value).expanduser().resolve()
        path_str = str(path_obj)
        instance.__dict__['original'] = os.path.normpath(path_str)

        path_parts = re.split(r'[/_. \-\\]', path_str)

        found_chars = pipe(
            path_parts,
            curried.filter(lambda c: len(c) == 1),
            curried.map(str.casefold),
            set,
            cur(set.__contains__),
            curried.keyfilter(d=cls.characters),
        )

        characters = found_chars.copy()

        sep_idx = path_str.rfind(os.sep)
        dot_idx = path_str.rfind('.', sep_idx + 1)

        if dot_idx == -1:
            string, ext = path_str, ''
        else:
            string, ext = path_str[:dot_idx], path_str[dot_idx + 1:]

        instance.__dict__['extension'] = ext

        if ext:
            ext = '.' + ext

        match = cls._match(string, characters)
        instance.__dict__['root'] = match[0]

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

        failed = [c for c in 'xy' if c not in found_chars]
        if failed:
            missing_fmt = ', '.join(failed)
            raise ValueError(f'Local path failed to parse {value!r}; missing required characters: {missing_fmt}')

        parts = []
        r = result[0]
        parts.extend([r[0], f'{{{r[1]}}}', r[2]])
        for r in list(result)[1:]:
            parts.extend([f'{{{r[1]}}}', r[2]])

        instance.__dict__['format'] = ''.join(parts) + ext

        instance.__dict__['root'] = parts[0].rsplit('/', 1)[0]
        if os.sep in parts[0]:
            instance.__dict__['root'] = parts[0].rsplit(os.sep, 1)[0]

        instance.__dict__['suffix'] = os.path.relpath(instance.format, instance.root)

        return instance

    @staticmethod
    def _match(string: str, characters: dict[str, str]):
        c = '|'.join(characters)
        pattern = rf"^(.*)({c})(.*)$"
        match = re.match(pattern, string, flags=re.IGNORECASE)
        if not match:
            return string, '', ''
        return match.groups()


if __name__ == '__main__':
    from dataclasses import dataclass


    @dataclass
    class MockTile:
        xtile: int
        ytile: int


    @dataclass
    class MockGrid:
        zoom: int = 19
        tiles = [MockTile(10, 20), MockTile(11, 21)]
        index = [0, 1]


    def run_tests():
        print("Running Local Source tests...")

        # Test Case 1: Standard /z/x/y structure
        path1 = 'data/tiles/z/x/y.png'
        local1 = Local.from_str(path1)
        print(f"\nTest 1: {path1}")
        print(f"  Extension: {local1.extension}")

        # Legacy behavior: extension does not include dot
        assert local1.extension == 'png'
        assert '{x}' in local1.format and '{y}' in local1.format and '{z}' in local1.format

        # Test Case 2: Underscore separator x_y_z
        path2 = '/var/data/tiles/19/x_y_z.jpg'
        local2 = Local.from_str(path2)
        print(f"\nTest 2: {path2}")
        print(f"  Format: {local2.format}")
        assert local2.extension == 'jpg'

        # Test Case 3: Missing Extension
        path3 = 'tiles/z/x/y'
        local3 = Local.from_str(path3)
        print(f"\nTest 3: {path3}")
        print(f"  Extension: '{local3.extension}'")
        assert local3.extension == ''

        # Test Case 4: Inverted Structure /y/x
        path4 = 'tiles/y/x.png'
        local4 = Local.from_str(path4)
        print(f"\nTest 4: {path4}")
        print(f"  Format: {local4.format}")

        # Test Case 5: 'accepts' Logic
        assert Local.accepts('foo/bar/x/y.png') is True
        assert Local.accepts('https://google.com/x/y.png') is False
        print("\nTest 5: Accepts logic passed")

        # Test Case 6: Files generation
        local1.grid = MockGrid()
        files = local1.files
        print(f"\nTest 6: Files generation for {path1}")
        print(files)
        assert len(files) == 2
        assert '10' in files[0] and '20' in files[0]

        print("\nAll tests passed successfully.")


    try:
        run_tests()
    except AssertionError as e:
        print(f"\n!!! TEST FAILED: {e}")
    except Exception as e:
        print(f"\n!!! ERROR: {e}")
