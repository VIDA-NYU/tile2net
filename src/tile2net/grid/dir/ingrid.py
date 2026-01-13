from __future__ import annotations

from .grid import Grid


class InGrid(
    Grid,
):
    @property
    def static(self):
        return self.ingrid.indir
