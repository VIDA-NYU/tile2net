from __future__ import annotations

import copy
from typing import *

import geopandas as gpd

from tile2net.grid.frame.namespace import namespace

if TYPE_CHECKING:
    from tile2net.grid.vecgrid.vecgrid import VecGrid


class Feature(
    namespace
):
    # todo: go back and investigate what the purpose of this module was and if it's still necessary
    basegrid: VecGrid

    def _get(
            self,
            instance: VecGrid,
            owner
    ) -> Self:
        self.basegrid = instance
        return copy.copy(self)

    locals().update(__get__=_get)

    def _ensure_network_column(self, key: str):
        if key not in self.basegrid:
            if 'geometry' in self.basegrid:
                crs = getattr(self.basegrid.geometry, 'crs', None)
            else:
                crs = None
            self.basegrid[key] = gpd.GeoSeries(
                [None] * len(self.basegrid),
                index=self.basegrid.index,
                crs=crs,
                name=key,
            )

    @property
    def polygons(self) -> gpd.GeoSeries:
        key = f'polygons.{self.__name__}'
        if key not in self.basegrid:
            self.basegrid._load_polygons()
            self._ensure_network_column(key)
        return self.basegrid.frame[key]

    @property
    def network(self) -> gpd.GeoSeries:
        key = f'lines.{self.__name__}'
        if key not in self.basegrid:
            self.basegrid._load_lines()
            self._ensure_network_column(key)
        return self.basegrid[key]
