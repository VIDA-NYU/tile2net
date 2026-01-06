from __future__ import annotations
import os
from .grid import Grid
import shutil

import pandas as pd

from .dir import Dir
from ..grid.grid import Grid


class InGrid(
    Grid,
):
    @property
    def infile(self):
        return self.grid.indir


