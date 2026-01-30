from __future__ import annotations

from typing import *

import pandas as pd

from tile2net.grid.frame.namespace import namespace
from .. import frame

if TYPE_CHECKING:
    from .grid import Grid


class VecTile(
    namespace,
):
    """
    Namespace for accessing vec-tile attributes aligned with in-tiles.

    See usage:
        >>> Grid.vectile
    """

    @property
    def grid(self) -> Grid:
        """Reference to parent Grid instance."""
        return self.instance

    @property
    def basegrid(self) -> Grid:
        """Reference to parent Grid instance."""
        return self.grid

    @property
    def dimension(self) -> int:
        """
        Pixel dimension of each vec-tile (width and height are equal).

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.dimension
            8192
        """
        return self.grid.vecgrid.dimension
