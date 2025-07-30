from __future__ import annotations

import copy
import functools
from pathlib import Path

import imageio.v3 as iio
import pandas as pd
import os
from .. import frame

if False:
    from tile2net.grid.grid.grid import Grid


def __get__(
        self: File,
        instance: Grid,
        owner: type[Grid],
) -> File:
    from .grid import Grid
    self.grid = instance
    return copy.copy(self)


class File(

):
    locals().update(
        __get__=__get__
    )
    grid: Grid = None

    def __init__(self, *args):
        ...

    def __set_name__(self, owner, name):
        self.__name__ = name

    @frame.column
    def infile(self) -> pd.Series:
        grid = self.grid
        key = 'file.infile'
        if key in grid:
            return grid[key]
        files = grid.indir.files(grid)
        grid[key] = files
        if (
                not grid.download
                and not files.map(os.path.exists).all()
        ):
            grid.download()
        return grid[key]

