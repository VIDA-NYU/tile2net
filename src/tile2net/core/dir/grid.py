from __future__ import annotations

from .basegrid import BaseGrid


class Grid(
    BaseGrid,
):
    @property
    def static(self):
        return self.basegrid.indir
