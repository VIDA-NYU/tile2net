from __future__ import annotations
import os
import shutil

import pandas as pd

from .dir import Dir
from ..tiles.tiles import Tiles


class InFile(
    Dir
):
    ...

    # def files(
    #         self,
    #         tiles: Tiles,
    #         dirname=''
    # ) -> pd.Series:
    #     return self.intiles.indir.files(tiles, dirname)
    #


class InTiles(
    Dir,
):
    # @InFile
    # def infile(self):
    #     ...

    @property
    def infile(self):
        return self.tiles.indir


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
