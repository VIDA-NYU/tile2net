from __future__ import annotations

import os
import os.path
from pathlib import Path

from .dir import Dir
from .ingrid import InGrid
from .namedir import NameDir


class Polygons(
    Dir
):

    @property
    def parquet(self) -> str:
        name = self.grid.name
        filename = os.path.join(self.dir, 'parquet', f'{name}.parquet')
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        return filename

    @property
    def preview(self) -> str:
        name = self.grid.name
        filename = os.path.join(self.dir, 'preview', f'{name}.png')
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        return filename


class Lines(
    Dir
):

    @property
    def parquet(self) -> str:
        name = self.grid.name
        filename = os.path.join(self.dir, 'parquet', f'{name}.parquet')
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        return filename

    @property
    def preview(self) -> str:
        name = self.grid.name
        filename = os.path.join(self.dir, 'preview', f'{name}.png')
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        return filename


class SourceDir(
    Dir
):

    @NameDir
    def namedir(self):
        grid = self.grid
        name = grid.cfg.name
        if name is None:
            name = grid.location
        format = os.path.join(
            self.dir,
            name,
            self.suffix
        )
        result = NameDir.from_format(format)
        return result

    @Lines
    def lines(self):
        ...

    @Polygons
    def polygons(self):
        ...

    @InGrid
    def ingrid(self):
        format = os.path.join(
            self.dir,
            'ingrid',
            self.suffix
        )
        result = InGrid.from_format(format)
        return result
