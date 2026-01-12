from __future__ import annotations

import os
import re
from abc import abstractmethod
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import Self

import pandas as pd

from tile2net.grid.source.exceptions import InvalidLocalPath
from tile2net.grid.source.source import Source


class Local(Source):
    characters = {
        c: i
        for i, c in enumerate('xyz')
    }

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
    def from_str(cls, value: str) -> Self:
        instance = cls()

        path_obj = Path(value).expanduser().resolve()
        path_str = str(path_obj)
        instance.original = os.path.normpath(path_str)

        path_parts = re.split(r'[/_. \-\\]', path_str)
        # filter for single-character parts, convert to lowercase, and collect as set
        single_char_parts = {
            part.casefold()
            for part in path_parts
            if len(part) == 1
        }

        # keep only characters that exist in both single_char_parts and cls.characters
        found_chars = {
            char: idx
            for char, idx in cls.characters.items()
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

    @staticmethod
    def _match(string: str, characters: dict[str, str]):
        """Match the first occurrence of any character in 'characters' within 'string'."""
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


    print("Testing Local Source...")
    print("=" * 80)

    # test home directory expansion and relative paths
    print("\n1. Testing home directory (~) and relative paths:")

    path_home = '~/tiles/x/y/z.png'
    local_home = Local.from_inferred(path_home)
    assert local_home.extension == 'png'
    assert '{x}' in local_home.format and '{y}' in local_home.format and '{z}' in local_home.format
    assert '~' not in local_home.format
    print(f"  ✓ '{path_home}' -> expanded to absolute path")
    print(f"    format={local_home.format}")

    path_rel = './tiles/x/y.png'
    local_rel = Local.from_inferred(path_rel)
    assert local_rel.extension == 'png'
    assert '{x}' in local_rel.format and '{y}' in local_rel.format
    assert not local_rel.format.startswith('.')
    print(f"  ✓ '{path_rel}' -> resolved to absolute path")

    path_rel2 = 'tiles/x/y.png'
    local_rel2 = Local.from_inferred(path_rel2)
    assert local_rel2.extension == 'png'
    assert os.path.isabs(local_rel2.format)
    print(f"  ✓ '{path_rel2}' -> resolved to absolute path")

    # test standard path structures
    print("\n2. Testing standard path structures:")

    path1 = 'data/tiles/z/x/y.png'
    local1 = Local.from_inferred(path1)
    assert local1.extension == 'png'
    assert '{x}' in local1.format and '{y}' in local1.format and '{z}' in local1.format
    print(f"  ✓ '{path1}' -> format={local1.format}")

    path2 = '/var/data/tiles/19/x_y_z.jpg'
    local2 = Local.from_inferred(path2)
    assert local2.extension == 'jpg'
    assert '{x}' in local2.format and '{y}' in local2.format and '{z}' in local2.format
    print(f"  ✓ '{path2}' -> extension={local2.extension}")

    path3 = 'tiles/y/x.png'
    local3 = Local.from_inferred(path3)
    assert local3.extension == 'png'
    assert '{x}' in local3.format and '{y}' in local3.format
    print(f"  ✓ '{path3}' -> inverted y/x structure")

    # test paths without z (z is optional)
    print("\n3. Testing paths without z (z is optional):")
    path_no_z1 = 'tiles/x/y.png'
    local_no_z1 = Local.from_inferred(path_no_z1)
    assert local_no_z1.extension == 'png'
    assert '{x}' in local_no_z1.format and '{y}' in local_no_z1.format
    assert '{z}' not in local_no_z1.format
    print(f"  ✓ '{path_no_z1}' -> no z coordinate")
    print(f"    format={local_no_z1.format}")

    path_no_z2 = 'data/imagery/x_y.jpg'
    local_no_z2 = Local.from_inferred(path_no_z2)
    assert local_no_z2.extension == 'jpg'
    assert '{x}' in local_no_z2.format and '{y}' in local_no_z2.format
    assert '{z}' not in local_no_z2.format
    print(f"  ✓ '{path_no_z2}' -> x_y pattern without z")

    # test missing extension
    print("\n4. Testing paths without extension:")
    path4 = 'tiles/z/x/y'
    local4 = Local.from_inferred(path4)
    assert local4.extension == ''
    print(f"  ✓ '{path4}' -> no extension")

    # test various separators
    print("\n5. Testing various separators:")
    path5 = 'tiles/x_y_z.png'
    local5 = Local.from_inferred(path5)
    assert '{x}' in local5.format and '{y}' in local5.format and '{z}' in local5.format
    print(f"  ✓ underscore separator: {path5}")

    path6 = 'tiles/x-y-z.png'
    local6 = Local.from_inferred(path6)
    assert '{x}' in local6.format and '{y}' in local6.format and '{z}' in local6.format
    print(f"  ✓ dash separator: {path6}")

    path7 = 'tiles/arst_x_y_z.png'
    local7 = Local.from_inferred(path7)
    assert '{x}' in local7.format and '{y}' in local7.format and '{z}' in local7.format
    print(f"  ✓ prefix with separator: {path7}")

    # test from_inferred (should delegate to from_str)
    print("\n6. Testing Local.from_inferred():")
    path8 = 'data/tiles/z/x/y.png'
    local8 = Local.from_inferred(path8)
    assert isinstance(local8, Local)
    assert local8.extension == 'png'
    print(f"  ✓ from_inferred delegates to from_str")

    # test files property
    print("\n8. Testing files generation:")
    local1.grid = MockGrid()
    files = local1.files
    assert len(files) == 2
    assert '10' in files[0] and '20' in files[0]
    assert '11' in files[1] and '21' in files[1]
    print(f"  ✓ Generated {len(files)} file paths")
    print(f"    {files[0]}")
    print(f"    {files[1]}")

    # test error cases
    print("\n9. Testing error cases:")
    try:
        Local.from_inferred('input/dir/x.png')
        assert False, "Should have raised InvalidLocalPath (missing y)"
    except InvalidLocalPath as e:
        print(f"  ✓ Missing 'y' raises InvalidLocalPath")

    try:
        Local.from_inferred('input/dir/y.png')
        assert False, "Should have raised InvalidLocalPath (missing x)"
    except InvalidLocalPath as e:
        print(f"  ✓ Missing 'x' raises InvalidLocalPath")

    try:
        Local.from_inferred('input/dir/xy.png')
        assert False, "Should have raised InvalidLocalPath (xy together)"
    except InvalidLocalPath as e:
        print(f"  ✓ Characters not separated raises InvalidLocalPath")

    try:
        Local.from_inferred('input/dir/a/b/c.png')
        assert False, "Should have raised InvalidLocalPath (no x/y)"
    except InvalidLocalPath as e:
        print(f"  ✓ No x/y characters raises InvalidLocalPath")

    try:
        Local.from_inferred('input/dir/z.png')
        assert False, "Should have raised InvalidLocalPath (only z, missing x and y)"
    except InvalidLocalPath as e:
        print(f"  ✓ Only 'z' (missing x and y) raises InvalidLocalPath")

    # test root and suffix extraction
    print("\n10. Testing path component extraction:")
    path9 = '/home/user/tiles/z/x/y.png'
    local9 = Local.from_inferred(path9)
    assert local9.root
    assert local9.suffix
    print(f"  ✓ root: {local9.root}")
    print(f"  ✓ suffix: {local9.suffix}")

    print("\n" + "=" * 80)
    print("All Local.from_inferred() and from_inferred() tests passed! ✓")
    print("=" * 80)
