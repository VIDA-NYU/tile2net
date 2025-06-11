from __future__ import annotations

import itertools
from typing import *

import geopandas as gpd
import pandas as pd
from geopandas import GeoSeries
from shapely import *

from ..fixed import GeoDataFrameFixed

if False:
    from .pednet import PedNet


def __get__(
        self,
        instance: PedNet,
        owner: type[PedNet]
) -> Self:
    if instance is None:
        result = self
    elif self.__name__ in instance.attrs:
        result = instance.attrs[self.__name__]
    else:
        result = instance.dissolve(by='f_type')
        instance.attrs[self.__name__] = result
    result._pednet = instance

    return self


class Features(
    GeoDataFrameFixed
):
    __name__ = 'features'
    _pednet = None

    def __set__(
            self,
            instance: PedNet,
            value: type[PedNet],
    ):
        value.__name__ = self.__name__
        instance.attrs[self.__name__] = value

    @property
    def above(self) -> GeoSeries[GeometryCollection]:
        """Unary union of all features above the given feature."""
        key = f'{self.__name__}.above'
        if key in self:
            return self[key]
        ordered = self.sort_values(self.z_order.name, ascending=False)
        geoms = ordered.geometry.values
        out = []
        cum = GeometryCollection()
        for g in geoms:
            out.append(cum)
            cum = cum.union(g)
        result = GeoSeries(out, index=ordered.index, crs=self.crs)
        self[key] = result
        result = self[key]
        return result

    @property
    def mutex(self) -> GeoSeries[GeometryCollection]:
        """Mutually exclusive geometries based on z_order"""
        key = f'{self.__name__}.mutex'
        if key in self:
            return self[key]
        result = self.geometry.difference(self.above)
        self[key] = result
        result = self[key]
        return result

    def clip(
            self,
            lines: gpd.GeoDataFrame,
    ):
        """Use the feature polygons to clip the centerlines."""
        return NotImplementedError

    @property
    def color(self) -> pd.Series:
        key = f'{self.__name__}.color'
        if key in self:
            return self[key]
        n = len(self)
        colors = (
            'red green blue orange purple brown gray cyan magenta '
            'yellow pink lime gold navy olive teal maroon coral '
            'turquoise violet indigo tan tomato silver'
        ).split()
        it = itertools.cycle(colors)
        result = list(itertools.islice(it, n))
        self[key] = pd.Series(result, index=self.index, dtype='string')
        result = self[key]
        return result
