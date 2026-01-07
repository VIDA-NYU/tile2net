from __future__ import annotations

from .grid import Grid


class InGrid(
    Grid,
):
    @property
    def infile(self):
        return self.grid.indir
