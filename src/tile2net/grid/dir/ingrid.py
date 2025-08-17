from __future__ import annotations
import os
import shutil

import pandas as pd

from .dir import Dir
from ..grid.grid import Grid


class InFile(
    Dir
):
    ...


class InGrid(
    Dir,
):
    # @InFile
    # def infile(self):
    #     ...

    @property
    def infile(self):
        return self.grid.indir


