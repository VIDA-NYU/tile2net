from __future__ import annotations

from typing import *

import geopandas as gpd
import numpy as np
import shapely
from centerline.geometry import Centerline

if False:
    from .pednet import PedNet


def __get__(
        self,
        instance: PedNet,
        owner: type[PedNet]
) -> Self:
    self.instance = instance
    self.owner = owner
    return self


class Dissolved:
    locals().update(
        __get__=__get__,
    )
    instance: PedNet = None
    __name__ = 'dissolved'

    def __init__(self, *args, **kwargs):
        ...

    @property
    def polygons(self) -> gpd.GeoDataFrame:
        geometry = self.instance.geometry.union_all()
        data = shapely.get_parts(geometry)
        crs = self.instance.crs
        geometry = gpd.GeoSeries(data, crs=crs)
        result = gpd.GeoDataFrame(geometry=geometry)
        result.index.name = 'idissolved'
        return result

    @property
    def centerlines(self) -> gpd.GeoDataFrame:
        df = self.polygons
        geometry = df.geometry
        it = (
            Centerline(poly).geometry
            for poly in geometry
        )
        multilines = np.fromiter(it, dtype=object, count=len(df))
        repeat = shapely.get_num_geometries(multilines)
        lines = shapely.get_parts(multilines)
        iloc = np.arange(len(repeat)).repeat(repeat)
        result = (
            self.polygons
            .iloc[iloc]
            .set_geometry(lines)
            .simplify(.1)
            .reset_index()
        )
        result.index.name = 'icent'
        return result

        # result = gpd.GeoSeries(data, index=df.index, crs=df.crs, name='geometry')
