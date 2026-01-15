from __future__ import annotations

import os
import os.path

import os
import os.path
from pathlib import Path

from .dir import Dir
from .grid import Grid


from .dir import Dir
from .grid import Grid
from .seggrid import SegGrid
from .vecgrid import VecGrid
from .basegrid import BaseGrid

class Polygons(
    Dir
):

    @property
    def parquet(self) -> str:
        name = self.basegrid.name
        filename = os.path.join(self.dir, 'parquet', f'{name}.parquet')
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        return filename

    @property
    def preview(self) -> str:
        name = self.basegrid.name
        filename = os.path.join(self.dir, 'preview', f'{name}.png')
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        return filename


class Network(
    Dir
):

    @property
    def parquet(self) -> str:
        name = self.basegrid.name
        filename = os.path.join(self.dir, 'parquet', f'{name}.parquet')
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        return filename

    @property
    def preview(self) -> str:
        name = self.basegrid.name
        filename = os.path.join(self.dir, 'preview', f'{name}.png')
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        return filename


class ProjectDir(
    BaseGrid,
):
    @VecGrid
    def vecgrid(self):
        template = os.path.join(
            self.dir,
            'vecgrid',
            self.suffix
        )
        result = VecGrid.from_template(template)
        return result

    @SegGrid
    def seggrid(self):
        template = os.path.join(
            self.dir,
            'seggrid',
            self.suffix
        )
        result = SegGrid.from_template(template)
        return result

    @Grid
    def grid(self):
        template = os.path.join(
            self.dir,
            'grid',
            self.suffix
        )
        result = Grid.from_template(template)
        return result

    @Network
    def network(self):
        ...

    @Polygons
    def polygons(self):
        ...

    @Grid
    def grid(self):
        return Grid.from_parent(self, 'grid')
