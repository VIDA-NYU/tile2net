from __future__ import annotations

from pathlib import Path
from typing import *

from tile2net.raster.tile_utils.topology import *
from .center import Center
from .dissolved import Dissolved
from .features import Features
from ..fixed import GeoDataFrameFixed


class PedNet(
    GeoDataFrameFixed,
):
    __name__ = 'pednet'

    @classmethod
    def from_polygons(
            cls,
            gdf: gpd.GeoDataFrame,
    ) -> Self:
        result = cls(gdf)
        return result

    @classmethod
    def from_parquet(
            cls,
            path: Union[str, Path],
    ) -> Self:
        """
        Load a PedNet from a parquet file.
        """
        result = (
            gpd.read_parquet(path)
            .pipe(cls.from_polygons)
        )
        return result

    @Dissolved
    def dissolved(self):
        # This code block is just semantic sugar and does not run.
        _ = self.dissolved.polygons
        _ = self.dissolved.centerlines

    @Features
    def features(self):
        # This code block is just semantic sugar and does not run.
        _ = self.features.polygons
        _ = self.features.centerlines

    @Center
    def center(self):
        # This code block is just semantic sugar and does not run.
        _ = self.center
        _ = self.center.sidewalk
        _ = self.center.crosswalk
