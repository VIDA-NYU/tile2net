import os
from os import PathLike
from os import fspath

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
    ...


    # @InFile
    # def infile(self):
    #     intiles = self.intiles.outdir.intiles
    #     format = os.path.join(
    #         intiles.dir,
    #         'infile',
    #         intiles.suffix,
    #     )
    #     result = InFile.from_format(format)
    #     return result
    #
