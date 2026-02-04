from __future__ import annotations

import os
import os.path
import os.path
from pathlib import Path

from .basegrid import BaseGrid
from .dir import Dir
from .ingrid import InGrid
from .seggrid import SegGrid
from .vecgrid import VecGrid


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
        return VecGrid.from_parent(self, 'vecgrid')

    @SegGrid
    def seggrid(self):
        return SegGrid.from_parent(self, 'seggrid')

    @InGrid
    def ingrid(self):
        return InGrid.from_parent(self, 'ingrid')

    @Network
    def network(self):
        ...

    @Polygons
    def polygons(self):
        ...

