import os

import pandas as pd
from tile2net.tiles.intiles.source import Source
from .dir import Dir, Dir


class InFile(
    Dir
):
    ...


class Indir(
    Dir
):

    @InFile
    def infile(self):
        format = os.path.join(
            self.dir,
            'infile',
            self.suffix,
        )
        result = InFile.from_format(format)
        return result
