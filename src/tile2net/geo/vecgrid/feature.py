from __future__ import annotations

import copy
from typing import Self, TYPE_CHECKING

import geopandas as gpd

from tile2net.core.frame.namespace import namespace

if TYPE_CHECKING:
    from tile2net.geo.vecgrid.vecgrid import VecGrid


class Feature(
    namespace
):
    # todo: go back and investigate what the purpose of this module was and if it's still necessary
    grid: VecGrid

    def _get(
            self,
            instance: VecGrid,
            owner
    ) -> Self:
        self.grid = instance
        return copy.copy(self)

    locals().update(__get__=_get)

    def _ensure_network_column(self, key: str):
        if key not in self.grid:
            if 'geometry' in self.grid:
                crs = getattr(self.grid.geometry, 'crs', None)
            else:
                crs = None
            self.grid[key] = gpd.GeoSeries(
                [None] * len(self.grid),
                index=self.grid.index,
                crs=crs,
                name=key,
            )

    @property
    def polygons(self) -> gpd.GeoSeries:
        key = f'polygons.{self.__name__}'
        if key not in self.grid:
            self.grid._load_polygons()
            self._ensure_network_column(key)
        return self.grid.frame[key]

    @property
    def network(self) -> gpd.GeoSeries:
        key = f'lines.{self.__name__}'
        if key not in self.grid:
            self.grid._load_lines()
            self._ensure_network_column(key)
        return self.grid[key]
