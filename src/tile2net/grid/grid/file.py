from __future__ import annotations

import copy
import os

from tile2net.grid.frame.namespace import namespace

if False:
    from tile2net.grid.frame import column
    from tile2net.grid.grid.grid import Grid

from .. import frame


class File(
    namespace
):
    @property
    def grid(self) -> Grid:
        return self.instance

    @frame.column
    def infile(self):
        grid = self.grid
        files = grid.indir.files(grid)
        if (
                not grid.download
                and not files.map(os.path.exists).all()
        ):
            grid.download()
        return files
