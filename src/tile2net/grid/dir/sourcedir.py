from __future__ import annotations

import os
import os.path
from pathlib import Path

from .dir import Dir
from .grid import Grid
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


class Network(
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
            name = grid
        # format = os.path.join(
        #     self.dir,
        #     name,
        #     self.suffix
        # )
        # result = NameDir.from_format(format)
        result = NameDir.from_template(self, name=name)
        return result

    @Network
    def network(self):
        ...

    @Polygons
    def polygons(self):
        ...

    @Grid
    def grid(self):
        return Grid.from_parent(self, 'grid', )
