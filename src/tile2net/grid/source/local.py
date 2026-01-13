from __future__ import annotations

import os
from abc import abstractmethod
from functools import *
from pathlib import Path
from typing import *

import pandas as pd
import pyarrow as pa

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
