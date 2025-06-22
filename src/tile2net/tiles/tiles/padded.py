from __future__ import annotations

from typing import *

from .tiles import Tiles


class Padded(
    Tiles
):

    @property
    def intiles(self):
        return self.instance.intiles

    @property
    def segtiles(self):
        return self.instance.segtiles

    @property
    def vectiles(self):
        return self.instance.vectiles

    @property
    def padded(self) -> Self:
        return self
