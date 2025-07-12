
from __future__ import annotations

from functools import *
from typing import *

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.ops
from geopandas import GeoDataFrame
from pandas import Series

from ..fixed import GeoDataFrameFixed
import pandas as pd

if False:
    from .pednet import PedNet
    import folium
    from .stubs import Stubs
    from .mintrees import Mintrees
    from .lines import Lines

"""
without magicpandas,
input: lines
output: nodes, aggregated

one to drop degree=2 noeds
one to extract node information
"""

def __get__(
        self: Nodes,
        instance: Lines,
        owner
) -> Nodes:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        lines = shapely.get_parts(instance.geometry)
        coords = shapely.get_coordinates(lines, include_z=False)
        npoints = shapely.get_num_points(lines)
        iline = np.arange(len(npoints)).repeat(npoints)
        unique, ifirst, repeat = np.unique(
            iline,
            return_counts=True,
            return_index=True
        )
        istop = ifirst + repeat
        ilast = istop - 1
        iloc = np.c_[ifirst, ilast].ravel()
        ends = coords[iloc]

        haystack = pd.MultiIndex.from_arrays(ends.T)
        needles = haystack.drop_duplicates()
        x = needles.get_level_values(0).values
        y = needles.get_level_values(1).values
        degree = (
            pd.Series(1, haystack)
            .groupby(level=[0, 1], sort=False)
            .sum()
            .loc[needles]
            .values
        )
        geometry = shapely.points(x, y)
        crs = instance.crs
        data = dict(
            degree=degree,
            x=x,
            y=y,
        )
        result = self.__class__(data, geometry=geometry, crs=crs)
        result.index.name = 'inode'

        instance.__dict__[self.__name__] = result

    result.lines = instance

    return result


class Nodes(
    GeoDataFrameFixed
):
    x: Series
    y: Series
    degree: Series
    lines: Lines
    locals().update(
        __get__=__get__,
    )
    __name__ = 'nodes'

    @property
    def lines(self):
        try:
            return self.attrs['lines']
        except KeyError as e:
            raise AttributeError(
                'Lines not set. Please set lines attribute before accessing Nodes.'
            ) from e

    @lines.setter
    def lines(self, value: Lines):
        from .lines import Lines
        if not isinstance(value, Lines):
            raise TypeError('lines must be an instance of Lines')
        self.attrs['lines'] = value

    @property
    def inode(self) -> pd.Index:
        if 'inode' in self.index.names:
            return self.index.get_level_values('inode')
        else:
            return self['inode']

    @property
    def tuple(self) -> pd.Series:
        key = 'tuple'
        if key in self:
            return self[key]
        edges = self.lines.edges
        result = (
            edges
            .reset_index()
            .groupby('start_inode', sort=False)
            ['start_iend']
            .apply(tuple)
            .loc[self.inode.values]
        )
        self[key] = result
        result = self[key]
        return result

    @property
    def edges(self):
        return self.lines.edges

    @property
    def iunion(self) -> pd.Series:
        key = 'iunion'
        if key in self:
            return self[key]
        polygons = self.lines.pednet.union
        geometry = self.geometry
        iloc, idissolved = polygons.sindex.nearest(
            geometry, return_distance=False
        )
        labels: pd.Index = geometry.index[iloc]
        assert not labels.has_duplicates

        result = (
            polygons
            .reset_index()
            .iunion
            .iloc[idissolved]
            .set_axis(labels)
        )
        self[key] = result
        result = self[key]
        return result

    @property
    def threshold(self) -> pd.Series:
        key = 'threshold'
        if key in self:
            return self[key]
        result = (
            self.lines.pednet.union.boundary
            .reindex(self.iunion)
            .distance(self.geometry, align=False)
            .mul(2)
            .values
        )
        self[key] = result
        result = self[key]
        return result

