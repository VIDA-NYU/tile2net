
from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import shapely.ops
from pandas import Series

from .. import frame
from ...grid.frame.framewrapper import FrameWrapper
from ...grid.frame.namespace import namespace

if False:
    from .lines import Lines

"""
without magicpandas,
input: lines
output: nodes, aggregated

one to drop degree=2 noeds
one to extract node information
"""

class Nodes(
    FrameWrapper,
):
    x: Series
    y: Series
    degree: Series
    lines: Lines
    
    def _get(
            self,
            instance: Lines,
            owner
    ) -> Nodes:
        self: Self = namespace._get(self, instance, owner)
        cache = instance.frame.__dict__
        key = self.__name__
        if instance is None:
            return self
        if key in cache:
            result = cache[key]
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
            result = self.from_frame(data, geometry=geometry, crs=crs, wrapper=self)
            result.index.name = 'inode'

            cache[key] = result

        result.lines = instance

        return result
        
    locals().update(
        __get__=_get,
    )
    __name__ = 'nodes'

    @property
    def lines(self):
        try:
            return self.frame.attrs['lines']
        except KeyError as e:
            raise AttributeError(
                'Lines not set. Please set lines attribute before accessing Nodes.'
            ) from e

    @lines.setter
    def lines(self, value: Lines):
        from .lines import Lines
        if not isinstance(value, Lines):
            raise TypeError('lines must be an instance of Lines')
        self.frame.attrs['lines'] = value

    @frame.column
    def inode(self) -> pd.Index:
        """Node index values."""
        if 'inode' in self.index.names:
            return self.index.get_level_values('inode')
        else:
            return self['inode']

    @frame.column
    def tuple(self) -> pd.Series:
        """Tuple of edge indices for each node."""
        edges = self.lines.edges
        result = (
            edges
            .reset_index()
            .groupby('start_inode', sort=False)
            ['start_iend']
            .apply(tuple)
            .loc[self.inode.values]
        )
        return result

    @frame.column
    def edges(self):
        """Get edges from lines."""
        return self.lines.edges

    @frame.column
    def iunion(self) -> pd.Series:
        """Union index for each node."""
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
        return result

    @frame.column
    def threshold(self) -> pd.Series:
        """Threshold distance for each node."""
        result = (
            self.lines.pednet.union.boundary
            .reindex(self.iunion)
            .distance(self.geometry, align=False)
            .mul(2)
            .values
        )
        return result

