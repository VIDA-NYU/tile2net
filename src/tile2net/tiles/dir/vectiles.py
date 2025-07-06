from __future__ import annotations
from .segtiles import SegTiles

import os
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import os

import datetime
import os.path

import pandas as pd
from pandas.tseries.holiday import USPresidentsDay

from .batchiterator import BatchIterator
from .dir import Dir, Dir, Dir, Dir


class Polygons(
    Dir
):
    ...


class Network(
    Dir
):
    ...



class VecTiles(
    SegTiles
):

    @Polygons
    def polygons(self):
        format = os.path.join(
            self.dir,
            'polygons',
            self.suffix,
        ).replace(self.extension, 'parquet')
        result = Polygons.from_format(format)
        return result

    @Network
    def network(self):
        format = os.path.join(
            self.dir,
            'network',
            self.suffix,
        ).replace(self.extension, 'parquet')
        result = Network.from_format(format)
        return result
