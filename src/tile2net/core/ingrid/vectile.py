from __future__ import annotations

from typing import *

import pandas as pd

from tile2net.core.frame.namespace import namespace
from .. import frame

if TYPE_CHECKING:
    from .ingrid import InGrid


class VecTile(
    namespace,
):
    """
    Namespace for accessing vec-tile attributes aligned with in-tiles.

    See usage:
        >>> InGrid.vectile
    """

    @property
    def ingrid(self) -> InGrid:
        """Reference to parent Grid instance."""
        return self.instance

    @property
    def grid(self) -> InGrid:
        """Reference to parent Grid instance."""
        return self.ingrid

    @property
    def dimension(self) -> int:
        """
        Pixel dimension of each vec-tile (width and height are equal).

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.vectile.dimension
            8192
        """
        return self.ingrid.vecgrid.dimension
