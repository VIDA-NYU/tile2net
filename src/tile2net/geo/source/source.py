from __future__ import annotations

import copy
from typing import *

import geopandas as gpd
import shapely

from tile2net.geo.basegrid.basegrid import BaseGrid
from tile2net.core.frame.weak import weak
from tile2net.core.source import source
from tile2net.core.source.exceptions import SourceParseError

if TYPE_CHECKING:
    from ..ingrid import InGrid


class Source(
    source.Source,
):
    zoom: int
    """XYZ zoom level for the source.
    Our model performs best with a zoom of at least 19.
    """

    @weak.property
    def ingrid(self) -> Optional[InGrid]:
        """The Grid instance this source is attached to."""
        return None

    @classmethod
    def from_grid(cls, value: BaseGrid) -> Self:
        from .remote import Remote
        bbox_geom = shapely.geometry.box(*value.frame.total_bounds)
        out = (
            gpd.GeoSeries([bbox_geom], crs=value.frame.crs)
            .to_crs(4326)
            .pipe(Remote.from_inferred)
        )
        return out

    @classmethod
    def from_inferred(cls, value) -> Self:
        """ Infer the appropriate Source (Local or Remote) from various input types. """
        from .remote import Remote
        from .local import Local

        if isinstance(value, Source):
            return copy.copy(value)

        try:
            out = Local.from_inferred(value)
        except SourceParseError:
            ...
        else:
            return out
        try:
            out = Remote.from_inferred(value)
        except SourceParseError:
            ...
        else:
            if not out.server:
                msg = f'Remote source must have a server defined: {value!r}'
                raise ValueError(msg)
            return out

        msg = f'Cannot infer source from value: {value!r}'
        raise ValueError(msg)
