from __future__ import annotations

from typing import *

from tile2net.grid.frame.namespace import namespace

if TYPE_CHECKING:
    from .vecgrid import VecGrid


class Padded(
    namespace
):

    instance: VecGrid
    @property
    def length(self) -> int:
        """
        Number of seg-tiles that comprise one dimension of a vec-tile
        after it has been padded a number of seg-tiles.
        """
        result = self.instance.length
        result += self.instance.cfg.vectorization.pad * 2
        return result

    @property
    def dimension(self) -> int:
        """
        Pixel dimension of each vec-tile after it has been
        padded a number of seg-tiles.
        """
        return self.instance.seggrid.dimension * self.length





