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
    @frame.column
    def infile(self):
        grid = self.grid.ingrid
        files = grid.indir.files(grid)
        if (
                not grid.download
                and not files.map(os.path.exists).all()
        ):
            grid.download()
        return files


    grid: Grid = None

    def __get(
            self: File,
            instance: Grid,
            owner: type[Grid],
    ) -> File:
        self.grid = instance
        return copy.copy(self)

    locals().update(
        __get__=__get
    )

    def __set_name__(self, owner, name):
        self.__name__ = name


