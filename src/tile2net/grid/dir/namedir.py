from __future__ import annotations

import os
import os.path

from ._dir import Dir
from .ingrid import InGrid
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
        result = VecGrid.from_format(format)
        return result

    @SegGrid
    def seggrid(self):
        format = os.path.join(
            self.dir,
            'seggrid',
            self.suffix
        )
        result = SegGrid.from_format(format)
        return result

    @InGrid
    def ingrid(self):
        format = os.path.join(
            self.dir,
            'ingrid',
            self.suffix
        )
        result = InGrid.from_format(format)
        return result