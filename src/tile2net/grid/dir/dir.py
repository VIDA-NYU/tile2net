from __future__ import annotations

import copy
import os
import re
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import *

import pandas as pd
import pyarrow as pa

from tile2net.grid.dir.exceptions import XYNotFoundError
from tile2net.grid.frame.namespace import namespace

if TYPE_CHECKING:
    from ..basegrid.basegrid import BaseGrid
    from ..grid.grid import Grid


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

    template: str
    """Directory template."""

    root: str
    """Root directory path."""

    original: str
    """Original path specification."""

    extension: str = 'png'
    """File extension. Png by default."""

    suffix: str
    """Path suffix."""

    dir: str
    """Directory path."""

    # @weak.property
    @cached_property
    def basegrid(self) -> Optional[Grid]:
        """The Grid instance this directory is attached to."""
        return

    """Mapping of coordinate characters to their indices."""
    characters = dict(
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
            out.basegrid = None
        elif isinstance(instance, Dir):
            cache = instance.__dict__
            name = self.__name__
            if name not in cache:
                if self.__wrapped__:
                    value = self.__wrapped__(instance)
                else:
                    value = self.from_parent(instance, name, self.extension)
                self.__set__(instance, value=value)
            out = cache[name]
            out.obj = instance.basegrid
        else:
            raise TypeError(instance)
        return out

    locals().update(__get__=_get)

    def __set__(self, instance: Dir, value) -> Self:
        if isinstance(value, str):
            value = self.from_template(value)
        if not isinstance(value, Dir):
            raise TypeError(value)
        dir = copy.copy(value)
        instance.__dict__[self.__name__] = dir
        dir.__name__ = self.__name__



    def __delete__(self, instance):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            ...

    def __bool__(self):
        return self.template is not None

    def __repr__(self):
        attrs = []
        for attr in 'template root original extension suffix dir'.split():
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
                + f''
        )
        if extension:
            suffix += f'.{extension.lstrip(".")}'
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
        indir.template = ''.join(parts)
        if ext:
            indir.template += ext
        indir.dir = parts[0].rsplit('/', 1)[0]
        indir.suffix = os.path.relpath(indir.original, indir.dir)
        return indir

    def files(
            self,
            grid: BaseGrid,
            dirname=''
    ) -> pd.Series:
        """Given the specifications of the grid, return a Series of filepaths for each tile."""
        suffix = (
            self.template
            .removeprefix(self.dir)
            .lstrip(os.sep)
        )
        template = os.path.join(self.dir, dirname, suffix)
        zoom = grid.zoom

        # most scalable series construction
        it = zip(grid.xtile.values, grid.ytile.values)
        obj = (
            template.format(z=zoom, y=ytile, x=xtile)
            for xtile, ytile in it
        )
        data = pa.array(obj, type=pa.string(), size=len(grid))
        mapper = {pa.string(): pd.ArrowDtype(pa.string())}.get
        out: pd.Series = data.to_pandas(types_mapper=mapper)
        out.name = self.__name__
        out.index = grid.index

        # makedirs
        unique_parents = (
            out.str.rsplit(os.sep, n=1)
            .list[0]
            .unique()
        )
        for d in unique_parents:
            os.makedirs(d, exist_ok=True)

        return out

    @classmethod
    def _match(
            cls,
            string: str,
            characters: dict[str, str]
    ):
        c = '|'.join(characters)
        pattern = rf"^(.*)({c})(.*)$"
        match = re.match(pattern, string)
        return match.groups()

    def __set_name__(self, owner, name):
        self.__name__ = name

if __name__ == '__main__':
    # template parsing: x/y/z structure
    d = Dir.from_template('/var/tiles/x/y/z.png')
    assert '{x}' in d.template
    assert '{y}' in d.template
    assert '{z}' in d.template
    assert d.extension == 'png'
    print(f'x/y/z.png: {d.template}')

    # template parsing: underscore separator
    d = Dir.from_template('/var/tiles/x_y_z.png')
    assert '{x}' in d.template
    assert '{y}' in d.template
    assert '{z}' in d.template
    assert d.extension == 'png'
    print(f'x_y_z.png: {d.template}')

    # template parsing: dash separator
    d = Dir.from_template('/var/tiles/x-y-z.jpg')
    assert '{x}' in d.template
    assert '{y}' in d.template
    assert '{z}' in d.template
    assert d.extension == 'jpg'
    print(f'x-y-z.jpg: {d.template}')

    # template parsing: no z (optional)
    d = Dir.from_template('/var/tiles/x/y.png')
    assert '{x}' in d.template
    assert '{y}' in d.template
    assert '{z}' not in d.template
    assert d.extension == 'png'
    print(f'x/y.png (no z): {d.template}')

    # template parsing: no extension
    d = Dir.from_template('/var/tiles/x/y/z')
    assert '{x}' in d.template
    assert '{y}' in d.template
    assert '{z}' in d.template
    assert d.extension == ''
    print(f'x/y/z (no ext): {d.template}')

    # template parsing: home directory expansion
    d = Dir.from_template('~/tiles/x/y.png')
    assert '~' not in d.template
    assert os.path.isabs(d.template)
    print(f'~/tiles/x/y.png expanded: {d.template}')

    # template parsing: relative path resolution
    d = Dir.from_template('./tiles/x/y.png')
    assert os.path.isabs(d.template)
    print(f'./tiles/x/y.png resolved: {d.template}')

    # from_parent: extension inherited
    parent = Dir.from_template('/var/tiles/x/y.png')
    child = Dir.from_parent(parent, 'subdir')
    assert child.extension == 'png'
    assert 'subdir' in child.dir
    print(f'from_parent (inherited ext): {child.template}')

    # from_parent: extension replaced
    parent = Dir.from_template('/var/tiles/x/y.png')
    child = Dir.from_parent(parent, 'subdir', 'jpg')
    assert child.extension == 'jpg'
    assert 'subdir' in child.dir
    print(f'from_parent (replaced ext): {child.template}')

    # from_parent: template structure preserved
    parent = Dir.from_template('/var/tiles/x_y.png')
    child = Dir.from_parent(parent, 'output')
    assert '_' in child.suffix or '/' in child.suffix
    print(f'from_parent (structure): parent.suffix={parent.suffix}, child.suffix={child.suffix}')

    # error: missing y
    try:
        Dir.from_template('/var/tiles/x.png')
        raise AssertionError('expected XYNotFoundError')
    except XYNotFoundError:
        print('missing y: XYNotFoundError raised')

    # error: missing x
    try:
        Dir.from_template('/var/tiles/y.png')
        raise AssertionError('expected XYNotFoundError')
    except XYNotFoundError:
        print('missing x: XYNotFoundError raised')

    # error: x and y not separated
    try:
        Dir.from_template('/var/tiles/xy.png')
        raise AssertionError('expected XYNotFoundError')
    except XYNotFoundError:
        print('xy together: XYNotFoundError raised')

    # force_xy=False allows missing x/y
    d = Dir.from_template('/var/tiles/z.png', force_xy=False)
    assert '{z}' in d.template
    assert '{x}' not in d.template
    assert '{y}' not in d.template
    print(f'force_xy=False: {d.template}')

    # __bool__
    d = Dir.from_template('/var/tiles/x/y.png')
    assert bool(d)
    d_empty = Dir()
    assert not bool(d_empty)
    print('__bool__: truthy when template set, falsy otherwise')

    print('all tests passed')
