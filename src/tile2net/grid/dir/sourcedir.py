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


class SourceDir(
    Dir
):

    @NameDir
    def namedir(self):
        """Subdirectory named after the particular directory.
        Uses the passed name, location, or bounding box.
        """
        grid = self.basegrid
        ymin, xmin, ymax, xmax = grid.lat_lon
        name = (
            grid.cfg.name
            or grid.location
            or f"{ymin:.2f},{xmin:.2f},{ymax:.2f},{xmax:.2f}"
        )
        result = NameDir.from_parent(self, name)
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
