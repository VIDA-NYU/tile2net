from __future__ import annotations

import os
from abc import abstractmethod
from functools import cached_property
from typing import Self

from tile2net.grid.dir.dir import Dir
from tile2net.grid.dir.exceptions import XYNotFoundError
from tile2net.grid.source.exceptions import InvalidLocalPath
from tile2net.grid.source.source import Source


class Local(
    Source,
    Dir
):
    @cached_property
    def dir(self) -> str:
        """Directory path, same as root for Local sources."""
        return self.root

    @cached_property
    @abstractmethod
    def zoom(self) -> int:
        """
        Default XYZ zoom level for local tiles.
        Can be overridden if zoom level is encoded in the path.
        """

    @cached_property
    @abstractmethod
    def dimension(self) -> int:
        """Default dimension of tiles in pixels."""
        # todo: we had a method to automatically infer this from file size?


    @classmethod
    def from_inferred(cls, value) -> Self:
        """
        Infer a Local source from various input types.
        Currently only supports string paths.
        """
        if isinstance(value, str):
            return cls.from_str(value)

        msg = f'Cannot infer Local source from type {type(value).__name__}: {value!r}'
        raise InvalidLocalPath(msg)

    @classmethod
    def from_str(cls, value: str) -> Self:
        """
        Parse a local file path string into a Local source.
        Delegates to Dir.from_format() and converts exceptions.
        """
        try:
            instance = cls.from_template(value, force_xy=True)
            return instance
        except XYNotFoundError as e:
            msg = f'Local path failed to parse {value!r}; missing required characters: {", ".join(e.missing)}'
            raise InvalidLocalPath(msg) from e


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
    assert '{x}' in local_home.template and '{y}' in local_home.template and '{z}' in local_home.template
    assert '~' not in local_home.template
    print(f"  ✓ '{path_home}' -> expanded to absolute path")
    print(f"    format={local_home.template}")

    path_rel = './tiles/x/y.png'
    local_rel = Local.from_inferred(path_rel)
    assert local_rel.extension == 'png'
    assert '{x}' in local_rel.template and '{y}' in local_rel.template
    assert not local_rel.template.startswith('.')
    print(f"  ✓ '{path_rel}' -> resolved to absolute path")

    path_rel2 = 'tiles/x/y.png'
    local_rel2 = Local.from_inferred(path_rel2)
    assert local_rel2.extension == 'png'
    assert os.path.isabs(local_rel2.template)
    print(f"  ✓ '{path_rel2}' -> resolved to absolute path")

    # test standard path structures
    print("\n2. Testing standard path structures:")

    path1 = 'data/tiles/z/x/y.png'
    local1 = Local.from_inferred(path1)
    assert local1.extension == 'png'
    assert '{x}' in local1.template and '{y}' in local1.template and '{z}' in local1.template
    print(f"  ✓ '{path1}' -> format={local1.template}")

    path2 = '/var/data/tiles/19/x_y_z.jpg'
    local2 = Local.from_inferred(path2)
    assert local2.extension == 'jpg'
    assert '{x}' in local2.template and '{y}' in local2.template and '{z}' in local2.template
    print(f"  ✓ '{path2}' -> extension={local2.extension}")

    path3 = 'tiles/y/x.png'
    local3 = Local.from_inferred(path3)
    assert local3.extension == 'png'
    assert '{x}' in local3.template and '{y}' in local3.template
    print(f"  ✓ '{path3}' -> inverted y/x structure")

    # test paths without z (z is optional)
    print("\n3. Testing paths without z (z is optional):")
    path_no_z1 = 'tiles/x/y.png'
    local_no_z1 = Local.from_inferred(path_no_z1)
    assert local_no_z1.extension == 'png'
    assert '{x}' in local_no_z1.template and '{y}' in local_no_z1.template
    assert '{z}' not in local_no_z1.template
    print(f"  ✓ '{path_no_z1}' -> no z coordinate")
    print(f"    format={local_no_z1.template}")

    path_no_z2 = 'data/imagery/x_y.jpg'
    local_no_z2 = Local.from_inferred(path_no_z2)
    assert local_no_z2.extension == 'jpg'
    assert '{x}' in local_no_z2.template and '{y}' in local_no_z2.template
    assert '{z}' not in local_no_z2.template
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
    assert '{x}' in local5.template and '{y}' in local5.template and '{z}' in local5.template
    print(f"  ✓ underscore separator: {path5}")

    path6 = 'tiles/x-y-z.png'
    local6 = Local.from_inferred(path6)
    assert '{x}' in local6.template and '{y}' in local6.template and '{z}' in local6.template
    print(f"  ✓ dash separator: {path6}")

    path7 = 'tiles/arst_x_y_z.png'
    local7 = Local.from_inferred(path7)
    assert '{x}' in local7.template and '{y}' in local7.template and '{z}' in local7.template
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
