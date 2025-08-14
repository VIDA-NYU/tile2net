from __future__ import annotations
from tile2net.grid.cfg.logger import logger
import geopandas as gpd

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
    lines: Lines

    @frame.column
    def x(self):
        ...

    @frame.column
    def y(self):
        ...

    @frame.column
    def degree(self):
        edges = self.lines.edges
        result = (
            edges.frame
            .groupby('start_inode', sort=False)
            .size()
        )
        return result

    def _get(
            self,
            instance: Lines,
            owner
    ) -> Nodes:
        self: Self = namespace._get(self, instance, owner)
        # cache = instance.frame.__dict__
        cache = instance.__dict__
        key = self.__name__
        if instance is None:
            return self
        if key in cache:
            result: Self = cache[key]
            # if result.instance is not instance:
            #     loc = result.inode.isin(instance.start_inode)
            #     loc |= result.inode.isin(instance.stop_inode)
            #     result = result.loc[loc]
            if result.instance is not instance:
                del cache[key]
                return self._get(instance, owner)
        elif (
                'start_inode' in instance.columns
                and 'stop_inode' in instance.columns
        ):
            start = (
                instance.frame
                ['start_inode start_x start_y'.split()]
            )
            stop = (
                instance.frame
                ['stop_inode stop_x stop_y'.split()]
            )
            names = 'inode x y'.split()

            result = (
                pd.concat([
                    start.set_axis(names, axis=1),
                    stop.set_axis(names, axis=1)
                ], ignore_index=True)
                .drop_duplicates('inode')
                .set_index('inode')
                .pipe(self.from_frame, wrapper=self)
            )
            cache[key] = result
        elif (
                'start_inode' in instance.columns
                or 'stop_inode' in instance.columns
        ):
            raise ValueError
        else:
            msg = f'Generating {owner.__name__}.{key}'
            logger.debug(msg)
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
            result = (
                gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
                .pipe(self.from_frame, wrapper=self)
            )
            result.index.name = 'inode'

            cache[key] = result

        result.instance = instance

        return result

    locals().update(
        __get__=_get,
    )
    __name__ = 'nodes'

    @property
    def lines(self) -> Lines:
        return self.instance

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
            edges.frame
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
        iloc, idissolved = polygons.frame.sindex.nearest(
            geometry, return_distance=False
        )
        labels: pd.Index = geometry.index[iloc]
        assert not labels.has_duplicates

        result = (
            polygons.frame
            .reset_index()
            .iunion
            .iloc[idissolved]
            .set_axis(labels)
        )
        return result

    @frame.column
    def threshold(self):
        """Threshold distance for each node."""
        lines = self.lines
        result = (
            self.lines.pednet.union.frame.boundary
            .reindex(self.iunion)
            .distance(self.geometry, align=False)
            .mul(2)
            .values
        )
        return result
