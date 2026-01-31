from __future__ import annotations

from typing import TYPE_CHECKING

import tile2net.geo.basegrid.file
import tile2net.grid.seggrid.file
from tile2net.grid.cfg.logger import logger

if TYPE_CHECKING:
    from .seggrid import SegGrid


class File(
    tile2net.grid.seggrid.file.File,
    tile2net.geo.basegrid.file.File,
):
    instance: SegGrid
    basegrid: SegGrid
