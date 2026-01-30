from __future__ import annotations

from tile2net.geo.seggrid import seggrid
from tile2net.xyz.basegrid import basegrid

class SegGrid(
    seggrid.SegGrid,
    basegrid.BaseGrid
):
    ...
