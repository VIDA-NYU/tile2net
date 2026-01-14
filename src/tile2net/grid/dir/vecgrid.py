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
        result = Polygons.from_template(format)
        return result

    @Network
    def network(self):
        format = os.path.join(
            self.dir,
            'network',
            self.suffix,
        )
        format = format + '.parquet'
        result = Network.from_template(format)
        return result

    @Network
    def curbs(self):
        format = os.path.joingg(
            self.dir,
            'curbs',
            self.suffix,
        )
        format = format + '.parquet'
        result = Network.from_template(format)
        return result


class VecGrid(Grid):
    @Dir
    def polygons(self):
        return Dir.from_parent(self, 'polygons', 'parquet')
