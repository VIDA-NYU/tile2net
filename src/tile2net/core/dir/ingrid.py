from __future__ import annotations

from .basegrid import BaseGrid


class InGrid(
    BaseGrid,
):
    @property
    def static(self):
        return self.basegrid.indir
