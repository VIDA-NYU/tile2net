from __future__ import annotations

import os
import os.path

from .dir import Dir
from .seggrid import SegGrid


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
        # try:
        #     format = format.replace(self.extension, 'parquet')
        # except AttributeError:
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
        # try:
        #     format = format.replace(self.extension, 'parquet')
        # except AttributeError:
        format = format + '.parquet'
        result = Network.from_format(format)
        return result
