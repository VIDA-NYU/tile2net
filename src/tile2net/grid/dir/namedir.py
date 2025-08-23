from __future__ import annotations

import hashlib
import os
import os.path
from pathlib import Path

import numpy as np

from .dir import Dir
from .ingrid import InGrid
from .seggrid import SegGrid
from .vecgrid import VecGrid

if False:
    import tile2net.grid.ingrid


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
