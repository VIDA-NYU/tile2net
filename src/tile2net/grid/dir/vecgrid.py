from __future__ import annotations
from .seggrid import SegGrid

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



class VecGrid(
    SegGrid
):

    @Polygons
    def polygons(self):
        format = os.path.join(
            self.dir,
            'polygons',
            self.suffix,
        )
        try:
            format = format.replace(self.extension, 'parquet')
        except AttributeError:
            format = format + '.parquet'
        result = Polygons.from_format(format)
        return result

    @Network
    def lines(self):
        format = os.path.join(
            self.dir,
            'lines',
            self.suffix,
        )
        try:
            format = format.replace(self.extension, 'parquet')
        except AttributeError:
            format = format + '.parquet'
        result = Network.from_format(format)
        return result
