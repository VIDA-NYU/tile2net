from __future__ import annotations

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
