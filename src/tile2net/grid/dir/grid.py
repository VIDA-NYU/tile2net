from __future__ import annotations

from ..basegrid.basegrid import BaseGrid


class Grid(
    BaseGrid,
):
    @property
    def static(self):
        return self.grid.indir
