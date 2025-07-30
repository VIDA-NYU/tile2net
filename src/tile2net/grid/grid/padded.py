from __future__ import annotations

from typing import *

from .grid import Grid


class Padded(
    Grid
):

    @property
    def ingrid(self):
        return self.instance.ingrid

    @property
    def seggrid(self):
        return self.instance.seggrid

    @property
    def vecgrid(self):
        return self.instance.vecgrid

    @property
    def padded(self) -> Self:
        return self
