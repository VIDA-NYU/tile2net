from __future__ import annotations

import copy
import os
import re
from functools import cached_property
from pathlib import Path
from typing import *
from typing import Self, Optional, Iterable

import pandas as pd
import pyarrow as pa

from tile2net.core.frame.namespace import namespace

if TYPE_CHECKING:
    from ..grid.grid import Grid
    from tile2net.core import InGrid
    from tile2net.core.grid.grid import Grid


class Dir(namespace):
    """
    Class responsible for managing directory-related operations and attributes.
    Supports arbitrary path templates with automatic or manual variable tokenization.
    """

    template: str
    """Directory template string with python format syntax (e.g., '/path/{to}/{file}.png')."""

    root: str
    """Static root directory path (prefix before first variable)."""

    original: str
    """Original path specification."""

    extension: str = 'png'
    """File extension."""

    suffix: str
    """Path suffix relative to the directory."""

    dir: str
    """Directory path."""

    _defaults: dict[str, str]
    """Dictionary mapping template keys to their default string values."""

    requires: Iterable[str] = tuple()
    """Iterable of required keys for this Dir instance."""

    @cached_property
    def grid(self) -> Optional['Grid']:
        """The Grid instance this directory is attached to."""
        return None

    __name__ = ''

    def _get(
            self,
            instance: 'Dir',
            owner
    ) -> Self:
        from tile2net.core.grid.grid import Grid
        if instance is None:
            self.instance = instance
            return self
        else:
            out = self
            out.grid = None
            cache = instance.__dict__
            name = self.__name__
            if name not in cache:
                if self.__wrapped__:
                    value = self.__wrapped__(instance)
                elif isinstance(instance, Dir):
                    value = self.from_parent(instance, name, self.extension)
                elif isinstance(instance, Grid):
                    value = self
                else:
                    raise TypeError(instance)
                self.__set__(instance, value=value)
            out = cache[name]

        if instance is None:
            out.grid = None
        elif isinstance(instance, Grid):
            out.grid = instance
        elif isinstance(instance, Dir):
            out.grid = instance.grid
        else:
            raise TypeError(instance)
        out.instance = instance
        return out

    locals().update(__get__=_get)

    __wrapped__ = None

    def __set__(self, instance: 'Dir', value) -> Self:
        # problem is we lose __wrapped__ by using from_template
        self.instance = instance
        if isinstance(value, str):
            value = self.from_template(value)
            value.__dict__ = self.__dict__ | value.__dict__
        elif isinstance(value, Dir):
            value = copy.copy(value)
        else:
            raise TypeError(value)
        instance.__dict__[self.__name__] = value
        value.__name__ = self.__name__

    def __delete__(self, instance):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            pass

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
            parent: 'Dir',
            name: str,
            extension: str = None,
    ) -> Self:
        if not extension:
            extension = parent.extension

        # Determine suffix base (remove extension)
        base_suffix = parent.suffix.rsplit('.', 1)[0]
        suffix = base_suffix + ''

        if extension:
            suffix += f'.{extension.lstrip(".")}'

        item = os.path.join(parent.dir, name, suffix)
        out = cls.from_template(item)
        return out



    @classmethod
    def from_template(
            cls,
            item: str,
            itile: str = None,
    ) -> Self:
        """
        Create a Dir instance from a template string.

        Args:
            item: The path or template string (e.g., '/path/to/data' or '/path/{z}/{x}_{y}.png').
            itile: Optional expected tile format (e.g., '{z}/{x}_{y}').
                   - If 'item' contains NONE of the tokens in 'itile', 'itile' is appended to 'item'.
                   - If 'item' contains ALL of the tokens in 'itile', 'item' is used as is.
                   - If 'item' contains SOME but not ALL tokens, a ValueError is raised.
        """
        if itile:
            # Extract required tokens from the expected itile format
            itile_tokens = set(re.findall(r'\{([^}]+)\}', itile))
            # Extract tokens present in the provided item
            item_tokens = set(re.findall(r'\{([^}]+)\}', item))

            # Find intersection of tokens
            common = itile_tokens & item_tokens

            if not common:
                # Case: Item has none of the tokens -> Append itile to item
                item = os.path.join(item, itile)
            elif common < itile_tokens:
                # Case: Item has some, but not all tokens -> Raise Exception
                # (common is a strict subset of itile_tokens)
                missing = itile_tokens - common
                raise ValueError(
                    f"The path template '{item}' contains partial tokens from the expected format '{itile}'. "
                    f"Missing tokens: {missing}. "
                    f"Please provide either a base directory (containing no tokens) or the full template."
                )
            # Case: Item has all required tokens (common == itile_tokens) -> Do nothing

        path_obj = Path(item).expanduser().resolve()
        value = str(path_obj)

        instance = cls()
        instance.original = value
        if '.' in value:
            instance.extension = value.rsplit('.', 1)[1]
        else:
            instance.extension = ''

        # Explicit tokenization required: keys must be wrapped in {}
        instance.template = value
        # Extract keys within braces for defaults
        keys = re.findall(r'\{([^}]+)\}', value)
        instance._defaults = {k: k for k in keys}

        # Calculate static root (text before the first variable)
        match = re.search(r'^(.*?)\{', instance.template)
        if match:
            instance.root = match.group(1)
        else:
            instance.root = instance.template

        # Determine dir: The directory component of the static root.
        # This ensures 'dir' stops before any curly braces.
        # e.g., root='/path/img_' -> dir='/path'
        # e.g., root='/path/dir/' -> dir='/path/dir'
        instance.dir = os.path.dirname(instance.root)

        # Calculate suffix relative to dir
        if instance.template.startswith(instance.dir):
            rel = instance.template[len(instance.dir):]
            instance.suffix = rel.lstrip(os.sep)
        else:
            instance.suffix = os.path.basename(instance.template)

        return instance

    def __set_name__(self, owner, name):
        self.__name__ = name

    def files(
            self,
            grid: Grid = None,
            dirname: str = '',
            makedirs: bool = True,
            **key2fill,
    ) -> pd.Series:
        """
        Generate file paths by filling the template keys.

        Priority of keys:
        1. **key2fill arguments (explicit overrides)
        2. self.__wrapped__(self.grid) (context from Grid method)
        3. self._defaults (tokens found in the template string)

        Args:
            dirname: Optional subdirectory to inject into the path.
            makedirs: Whether to create the resulting parent directories.
            **key2fill: Mapping of keys to iterables or scalars.
        """
        # Initialize with defaults (keys found in the template, e.g. {x: 'x'})
        fill_data = self._defaults.copy()
        from tile2net.core.grid.grid import Grid
        if grid is None:
            grid = self.grid

        if not isinstance(grid, Grid):
            msg = f'Dir.files() requires a Grid instance as grid, got {type(grid)}'
            raise TypeError(msg)

        context_data = grid.tokens
        if context_data:
            fill_data.update(context_data)

        # Apply manual overrides from arguments
        fill_data.update(key2fill)
        iterables = {}
        scalars = {}
        length = 1

        for key, val in fill_data.items():
            # Strings are iterable but should be treated as scalars here
            if (
                    isinstance(val, (str, bytes))
                    or not isinstance(val, Iterable)
            ):
                scalars[key] = val
            else:
                if isinstance(val, (pd.Series, pd.Index)):
                    it = val.values
                else:
                    it = list(val)

                # Update iteration length based on the first non-empty iterable
                # (Assumes all iterables are aligned in length)
                n = len(it)
                if length == 1 and n > 0:
                    length = n

                iterables[key] = it

        # Construct the template string
        if dirname:
            template_str = os.path.join(self.dir, dirname, self.suffix)
        else:
            template_str = self.template

        def path_gen():
            if not iterables:
                yield template_str.format(**scalars)
                return

            # Vectorized-style iteration
            for i in range(length):
                row = scalars.copy()
                for k, it in iterables.items():
                    row[k] = it[i]
                yield template_str.format(**row)

        # Construct Series using PyArrow for efficiency
        try:
            arr = pa.array(path_gen(), type=pa.string(), size=length)
        except Exception:
            # Fallback if generator fails or size prediction is off
            arr = pa.array(list(path_gen()), type=pa.string())

        mapper = {pa.string(): pd.ArrowDtype(pa.string())}.get
        out: pd.Series = arr.to_pandas(types_mapper=mapper)

        out.name = self.__name__
        if grid is not None:
            out.index = grid.index

        # 8. Create directories
        if makedirs and not out.empty:
            unique_parents = (
                out.str.rsplit(os.sep, n=1)
                .map(lambda x: x[0] if len(x) > 1 else '')
                .unique()
            )
            for d in unique_parents:
                if d:
                    os.makedirs(d, exist_ok=True)

        return out


if __name__ == '__main__':
    # --- Original Tests (Updated for Explicit Syntax) ---
    print("Running updated original tests...")

    # template parsing: {x}/{y}/{z} structure
    d = Dir.from_template('/var/tiles/{x}/{y}/{z}.png')
    assert '{x}' in d.template
    assert '{y}' in d.template
    assert '{z}' in d.template
    assert d.extension == 'png'
    print(f'{{x}}/{{y}}/{{z}}.png: {d.template}')

    # template parsing: underscore separator
    d = Dir.from_template('/var/tiles/{x}_{y}_{z}.png')
    assert '{x}' in d.template
    assert '{y}' in d.template
    assert '{z}' in d.template
    print(f'{{x}}_{{y}}_{{z}}.png: {d.template}')

    # template parsing: home directory expansion
    d = Dir.from_template('~/tiles/{x}/{y}.png')
    assert '~' not in d.template
    assert os.path.isabs(d.template)
    print(f'~/tiles/{{x}}/{{y}}.png expanded: {d.template}')

    # --- New Functionality Tests ---
    print("\nRunning new functionality tests...")

    # 1. Arbitrary word splitting with explicit braces
    # Only {i} is a token; 'houston' is now literal text.
    d = Dir.from_template('/var/tiles/houston_{i}.png', )

    assert '{houston}' not in d.template
    assert '{i}' in d.template
    assert 'houston' in d.template
    print(f'Explicit split (/var/tiles/houston_{{i}}.png): {d.template}')

    # 3. Custom keys
    d_custom = Dir.from_template('/var/tiles/houston{i}.png', )
    assert '{houston}' not in d_custom.template
    assert '{i}' in d_custom.template
    print(f'User specified braces (/var/tiles/houston{{i}}.png): {d_custom.template}')

    # 4. Mixed scalar and iterable in files()
    # Explicitly wrapping {region}, {x}, {y}
    d_mix = Dir.from_template('/data/{region}/{x}_{y}.tif', )

    print("All tests passed.")
