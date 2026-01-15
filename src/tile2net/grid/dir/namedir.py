from __future__ import annotations

import os
import os.path

from .dir import Dir
from .grid import Grid
from .seggrid import SegGrid
from .vecgrid import VecGrid


class NameDir(
    Dir
):
    @VecGrid
    def vecgrid(self):
        format = os.path.join(
            self.dir,
            'vecgrid',
            self.suffix
        )
        result = VecGrid.from_template(format)
        return result

    @SegGrid
    def seggrid(self):
        format = os.path.join(
            self.dir,
            'seggrid',
            self.suffix
        )
        result = SegGrid.from_template(format)
        return result

    @Grid
    def grid(self):
        format = os.path.join(
            self.dir,
            'grid',
            self.suffix
        )
        result = Grid.from_template(format)
        return result
