from __future__ import annotations

from typing import *

from tile2net.geo.grid import filled
from tile2net.geo.ingrid.ingrid import InGrid

if TYPE_CHECKING:
    from tile2net.core.seggrid.seggrid import SegGrid


class Filled(
    filled.Filled,
    InGrid,
):
    """
    Grid extension that fills in missing tiles to align with SegGrid boundaries.

    Ensures complete coverage by adding tiles implicated by SegGrid padding.
    For example, if SegGrid requires 2x2 in-tiles but Grid is 1x1, this fills
    the missing 3 tiles.

    Handles lazy-loading of filled grid with padding:
    >>> Filled._get
    """
    instance: InGrid

    def _get(
            self,
            instance: InGrid,
            owner,
    ) -> Self:
        """
        Lazy-load factory method for accessing Filled from Grid

        Automatically expands the grid to include all tiles needed for proper
        segmentation with padding. Rescales to seggrid scale, adds padding,
        then rescales back to original grid scale.

        Returns:
            Filled instance with complete tile coverage for segmentation

        Example:
            >>> grid: InGrid
            >>> grid.filled
            Filled Grid with additional boundary tiles
        """
        if instance is None:
            return self
        # instance = instance.grid
        # self.instance = instance
        cache = instance.frame.__dict__
        key = self.__name__

        if key in cache:
            result = cache[key]
        else:
            seggrid = instance.seggrid
            tiles = (
                instance.cfg.segmentation.pad
                .__truediv__(instance.dimension)
                .__ceil__()
            )
            result = (
                instance
                .to_scale(seggrid.scale)
                .to_padding(tiles=tiles)
                .to_scale(instance.scale)
                .pipe(self.__class__.from_wrapper)
            )
            assert isinstance(result, self.__class__)

            result.__dict__.update(instance.__dict__)
            result.instance = instance
            instance.frame.__dict__[self.__name__] = result
        return result

    locals().update(__get__=_get)

    @property
    def seggrid(self) -> SegGrid:
        return self.instance.seggrid

    @property
    def filled(self):
        return self.instance.filled

    @property
    def ingrid(self) -> InGrid:
        return self.instance.ingrid
