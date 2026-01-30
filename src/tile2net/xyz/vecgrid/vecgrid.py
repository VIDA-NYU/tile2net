from __future__ import annotations

from tile2net.geo.vecgrid import vecgrid
from tile2net.xyz.basegrid import basegrid


class VecGrid(
    vecgrid.VecGrid,
    basegrid.BaseGrid,
):
    ...
