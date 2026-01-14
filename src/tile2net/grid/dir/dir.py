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
import pyarrow as pa

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
    def template(self) -> str:
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

    @cached_property
    def characters(self) -> dict[str, int]:
        """Mapping of coordinate characters to their indices."""
        return dict(
            x=0,
            y=1,
            z=2,
        )

    def _get(
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

    locals().update(__get__=_get)

    def __set__(self, instance: Dir, value) -> Self:
        if isinstance(value, str):
            value = self.from_template(value)
        if not isinstance(value, Dir):
            raise TypeError(value)
        instance.__dict__[self.__name__] = copy.copy(value)

    def __delete__(self, instance):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            ...

    def __bool__(self):
        return self.template is not None

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
        out = cls.from_template(item)
        return out

    @classmethod
    def from_template(
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
        indir.template = ''.join(parts) + (ext if ext else '')
        indir.dir = parts[0].rsplit('/', 1)[0]
        indir.suffix = os.path.relpath(indir.original, indir.dir)
        return indir

    def with_template(
            self,
            template: str = None
    ):
        if template is None:
            template = self.ingrid.cfg.template
        format = os.path.join(
            self.dir,
            template,
        )
        result = self.__class__.from_template(format)
        return result

    def files(
            self,
            grid: Grid,
            dirname=''
    ) -> pd.Series:
        suffix = (
            self.template
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


if __name__ == '__main__':
    import dataclasses
    from dataclasses import dataclass

    @dataclass
    class ConcreteDir(Dir):
        """Concrete implementation of Dir for testing."""
        template: str = None
        root: str = None
        original: str = None
        extension: str = None
        suffix: str = None
        dir: str = None
        ingrid: InGrid = None

        def __post_init__(self):
            for field in dataclasses.fields(self):
                if field.name not in self.__dict__:
                    setattr(self, field.name, field.default)

    @dataclass
    class MockTile:
        xtile: int
        ytile: int

    @dataclass
    class MockGrid:
        zoom: int = 19
        tiles = [MockTile(10, 20), MockTile(11, 21)]
        index = [0, 1]
        xtile = pd.Series([10, 11], index=index)
        ytile = pd.Series([20, 21], index=index)

    print("Testing Dir class...")
    print("=" * 80)

    # test characters property
    print("\n1. Testing characters property:")
    test_dir = ConcreteDir()
    assert test_dir.characters == {'x': 0, 'y': 1, 'z': 2}
    print("  ✓ characters property returns correct mapping")

    # test home directory expansion and relative paths
    print("\n2. Testing home directory (~) and relative paths:")
    path_home = '~/tiles/x/y/z.png'
    dir_home = ConcreteDir.from_template(path_home)
    assert dir_home.extension == 'png'
    assert '{x}' in dir_home.template and '{y}' in dir_home.template and '{z}' in dir_home.template
    assert '~' not in dir_home.template
    print(f"  ✓ '{path_home}' -> expanded to absolute path")
    print(f"    template={dir_home.template}")

    path_rel = './tiles/x/y.png'
    dir_rel = ConcreteDir.from_template(path_rel)
    assert dir_rel.extension == 'png'
    assert '{x}' in dir_rel.template and '{y}' in dir_rel.template
    assert not dir_rel.template.startswith('.')
    print(f"  ✓ '{path_rel}' -> resolved to absolute path")

    path_rel2 = 'tiles/x/y.png'
    dir_rel2 = ConcreteDir.from_template(path_rel2)
    assert dir_rel2.extension == 'png'
    assert os.path.isabs(dir_rel2.template)
    print(f"  ✓ '{path_rel2}' -> resolved to absolute path")

    # test standard path structures
    print("\n3. Testing standard path structures:")
    path1 = 'data/tiles/z/x/y.png'
    dir1 = ConcreteDir.from_template(path1)
    assert dir1.extension == 'png'
    assert '{x}' in dir1.template and '{y}' in dir1.template and '{z}' in dir1.template
    print(f"  ✓ '{path1}' -> template={dir1.template}")

    path2 = '/var/data/tiles/19/x_y_z.jpg'
    dir2 = ConcreteDir.from_template(path2)
    assert dir2.extension == 'jpg'
    assert '{x}' in dir2.template and '{y}' in dir2.template and '{z}' in dir2.template
    print(f"  ✓ '{path2}' -> extension={dir2.extension}")

    path3 = 'tiles/y/x.png'
    dir3 = ConcreteDir.from_template(path3)
    assert dir3.extension == 'png'
    assert '{x}' in dir3.template and '{y}' in dir3.template
    print(f"  ✓ '{path3}' -> inverted y/x structure")

    # test paths without z (z is optional)
    print("\n4. Testing paths without z (z is optional):")
    path_no_z1 = 'tiles/x/y.png'
    dir_no_z1 = ConcreteDir.from_template(path_no_z1)
    assert dir_no_z1.extension == 'png'
    assert '{x}' in dir_no_z1.template and '{y}' in dir_no_z1.template
    assert '{z}' not in dir_no_z1.template
    print(f"  ✓ '{path_no_z1}' -> no z coordinate")
    print(f"    template={dir_no_z1.template}")

    path_no_z2 = 'data/imagery/x_y.jpg'
    dir_no_z2 = ConcreteDir.from_template(path_no_z2)
    assert dir_no_z2.extension == 'jpg'
    assert '{x}' in dir_no_z2.template and '{y}' in dir_no_z2.template
    assert '{z}' not in dir_no_z2.template
    print(f"  ✓ '{path_no_z2}' -> x_y pattern without z")

    # test missing extension
    print("\n5. Testing paths without extension:")
    path4 = 'tiles/z/x/y'
    dir4 = ConcreteDir.from_template(path4)
    assert dir4.extension == ''
    print(f"  ✓ '{path4}' -> no extension")

    # test various separators
    print("\n6. Testing various separators:")
    path5 = 'tiles/x_y_z.png'
    dir5 = ConcreteDir.from_template(path5)
    assert '{x}' in dir5.template and '{y}' in dir5.template and '{z}' in dir5.template
    print(f"  ✓ underscore separator: {path5}")

    path6 = 'tiles/x-y-z.png'
    dir6 = ConcreteDir.from_template(path6)
    assert '{x}' in dir6.template and '{y}' in dir6.template and '{z}' in dir6.template
    print(f"  ✓ dash separator: {path6}")

    path7 = 'tiles/arst_x_y_z.png'
    dir7 = ConcreteDir.from_template(path7)
    assert '{x}' in dir7.template and '{y}' in dir7.template and '{z}' in dir7.template
    print(f"  ✓ prefix with separator: {path7}")

    # test from_parent
    print("\n7. Testing from_parent():")
    parent = ConcreteDir.from_template('data/tiles/x/y.png')
    child = ConcreteDir.from_parent(parent, 'subdir', 'jpg')
    assert 'subdir' in child.dir
    assert child.extension == 'jpg'
    print(f"  ✓ from_parent creates child dir: {child.dir}")
    print(f"    template={child.template}")

    # test from_parent inheriting extension
    child2 = ConcreteDir.from_parent(parent, 'another')
    assert child2.extension == 'png'
    print(f"  ✓ from_parent inherits extension when not specified")

    # test files generation
    print("\n8. Testing files() method:")
    dir1_copy = ConcreteDir.from_template('data/tiles/x/y.png')
    grid = MockGrid()
    files = dir1_copy.files(grid)
    assert len(files) == 2
    assert '10' in files[0] and '20' in files[0]
    assert '11' in files[1] and '21' in files[1]
    print(f"  ✓ Generated {len(files)} file paths")
    print(f"    {files[0]}")
    print(f"    {files[1]}")

    # test error cases
    print("\n9. Testing error cases:")
    try:
        ConcreteDir.from_template('input/dir/x.png')
        assert False, "Should have raised XYNotFoundError (missing y)"
    except XYNotFoundError as e:
        assert 'y' in e.args[0]
        print(f"  ✓ Missing 'y' raises XYNotFoundError")

    try:
        ConcreteDir.from_template('input/dir/y.png')
        assert False, "Should have raised XYNotFoundError (missing x)"
    except XYNotFoundError as e:
        assert 'x' in e.args[0]
        print(f"  ✓ Missing 'x' raises XYNotFoundError")

    try:
        ConcreteDir.from_template('input/dir/xy.png')
        assert False, "Should have raised XYNotFoundError (xy together)"
    except XYNotFoundError as e:
        print(f"  ✓ Characters not separated raises XYNotFoundError")

    try:
        ConcreteDir.from_template('input/dir/a/b/c.png')
        assert False, "Should have raised XYNotFoundError (no x/y)"
    except XYNotFoundError as e:
        print(f"  ✓ No x/y characters raises XYNotFoundError")

    try:
        ConcreteDir.from_template('input/dir/z.png')
        assert False, "Should have raised XYNotFoundError (only z, missing x and y)"
    except XYNotFoundError as e:
        assert 'x' in e.args[0] and 'y' in e.args[0]
        print(f"  ✓ Only 'z' (missing x and y) raises XYNotFoundError")

    # test force_xy=False
    print("\n10. Testing force_xy=False:")
    path_only_z = 'tiles/z.png'
    dir_only_z = ConcreteDir.from_template(path_only_z, force_xy=False)
    assert '{z}' in dir_only_z.template
    assert '{x}' not in dir_only_z.template and '{y}' not in dir_only_z.template
    print(f"  ✓ force_xy=False allows paths without x/y")

    # test root and suffix extraction
    print("\n11. Testing path component extraction:")
    path9 = '/home/user/tiles/z/x/y.png'
    dir9 = ConcreteDir.from_template(path9)
    assert dir9.root
    assert dir9.suffix
    print(f"  ✓ root: {dir9.root}")
    print(f"  ✓ suffix: {dir9.suffix}")

    # test __bool__
    print("\n12. Testing __bool__:")
    dir_with_template = ConcreteDir.from_template('tiles/x/y.png')
    assert bool(dir_with_template)
    print(f"  ✓ Dir with template is truthy")

    dir_without_template = ConcreteDir()
    assert not bool(dir_without_template)
    print(f"  ✓ Dir without template is falsy")

    # test __repr__
    print("\n13. Testing __repr__:")
    dir_repr = ConcreteDir.from_template('data/tiles/x/y.png')
    repr_str = repr(dir_repr)
    assert 'ConcreteDir' in repr_str
    print(f"  ✓ __repr__ works: {repr_str[:60]}...")

    print("\n" + "=" * 80)
    print("All Dir tests passed! ✓")
    print("=" * 80)
