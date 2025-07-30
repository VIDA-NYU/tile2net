import os
from os import PathLike
from os import fspath

import pandas as pd
from tile2net.grid.ingrid.source import Source
from .dir import Dir, Dir


class InFile(
    Dir
):
    ...



class Indir(
    Dir
):
    ...


    # @InFile
    # def infile(self):
    #     ingrid = self.ingrid.outdir.ingrid
    #     format = os.path.join(
    #         ingrid.dir,
    #         'infile',
    #         ingrid.suffix,
    #     )
    #     result = InFile.from_format(format)
    #     return result
    #
