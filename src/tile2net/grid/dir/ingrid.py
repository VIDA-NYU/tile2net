from __future__ import annotations

from ..basegrid.basegrid import BaseGrid


class InGrid(
    BaseGrid,
):
    @property
    def static(self):
        return self.ingrid.indir
