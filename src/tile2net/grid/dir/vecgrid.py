from __future__ import annotations

import os
import os.path

from .dir import Dir
from .grid import Grid


class Polygons(
    Dir
):
    ...


class Network(
    Dir
):
    ...


class VecGrid(
    Grid
):

    @Polygons
    def polygons(self):
        format = os.path.join(
            self.dir,
            'polygons',
            self.suffix,
        )
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
        format = format + '.parquet'
        result = Network.from_format(format)
        return result

    @Network
    def curbs(self):
        format = os.path.join(
            self.dir,
            'curbs',
            self.suffix,
        )
        format = format + '.parquet'
        result = Network.from_format(format)
        return result
