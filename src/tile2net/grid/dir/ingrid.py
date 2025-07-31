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


# def cleanup(self):
#     """
#     Cleanup all files and subdirectories from this directory
#     except for the infile directory.
#     """
#     dir = self.dir
#     exclude = [self.infile.dir]

    def cleanup(self):
        base_dir: str = os.path.abspath(self.dir)
        exclude = [self.infile.dir]
        self._cleanup(base_dir, exclude)
