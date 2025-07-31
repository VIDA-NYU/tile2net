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
        self: Edges,
        instance: Lines,
        owner
) -> Edges:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        cols = 'iline geometry start_inode stop_inode start_iend stop_iend'.split()
        _ = (
            instance.start_inode, instance.stop_inode,
            instance.start_iend, instance.stop_iend,
        )
        lines = (
            instance
            .reset_index()
            [cols]
        )
        reverse = (
            lines
            .set_geometry(lines.reverse())
            .assign(
                stop_inode=instance.start_inode.values,
                start_inode=instance.stop_inode.values,
                stop_iend=instance.start_iend.values,
                start_iend=instance.stop_iend.values,
                start_x=instance.stop_x.values,
                start_y=instance.stop_y.values,
                stop_x=instance.start_x.values,
                stop_y=instance.start_y.values,
            )
        )
        concat = lines, reverse
        result = (
            pd.concat(concat, )
            .set_index('start_iend')
            .pipe(self.__class__)
        )
        instance.__dict__[self.__name__] = result

    result.lines = instance

    return result


class Edges(
    GeoDataFrameFixed,
):
    stop_iend: pd.Series
    iline: pd.Series
    start_x: pd.Series
    start_y: pd.Series
    stop_x: pd.Series
    stop_y: pd.Series
    start_inode: pd.Series
    stop_inode: pd.Series

    locals().update(
        __get__=__get__,
    )
    __name__ = 'edges'

    @property
    def lines(self) -> Lines:
        try:
            return self.attrs['lines']
        except KeyError as e:
            raise AttributeError(
                'Lines not set. Please set lines attribute before accessing Nodes.'
            ) from e

    @lines.setter
    def lines(self, value: Lines):
        self.attrs['lines'] = value

    @property
    def nodes(self):
        return self.lines.nodes

    @property
    def start_iend(self) -> pd.Index:
        key = 'start_iend'
        if key in self.index.names:
            return self.index.get_level_values(key)
        else:
            return self[key]

    @property
    def start_degree(self) -> pd.Series:
        key = 'start_degree'
        if key in self:
            return self[key]
        result = (
            self.lines.nodes.degree
            .loc[self.start_inode]
            .values
        )
        self[key] = result
        result = self[key]
        return result

    @property
    def start_shared_iend(self) -> pd.Series:
        key = 'start_shared_iend'
        if key in self:
            return self[key]
        other = self.start_other_iend.astype('UInt32')
        loc = other.notna()
        try:
            _ = self.feature
        except KeyError:
            ...
        else:
            loc &= (
                self.feature
                .reindex(other, fill_value='')
                .eq(self.feature.values)
                .values
            )

        self[key] = other.where(loc)
        result = self[key]
        return result

    @property
    def stop_shared_iend(self) -> pd.Series:
        key = 'stop_shared_iend'
        if key in self:
            return self[key]
        other = self.stop_other_iend.astype('UInt32')
        loc = other.notna()
        try:
            _ = self.feature
        except KeyError:
            ...
        else:
            loc &= (
                self.feature
                .reindex(other, fill_value='')
                .eq(self.feature.values)
                .values
            )

        self[key] = other.where(loc)
        result = self[key]
        return result

    @property
    def start_other_iend(self) -> pd.Series:
        """Other iend of the node at self.start_iend"""
        key = 'start_other_iend'
        if key in self:
            return self[key]
        loc = self.start_degree == 2
        edges: Edges = self.loc[loc]
        groupby = (
            edges
            .reset_index()
            .groupby(edges.start_inode.values, sort=False)
        )
        if not len(groupby):
            result = pd.Series(dtype='UInt32')
            self[key] = result
            result = self[key]
            return result
        groupby.size().max()
        assert groupby.size().max() <= 2
        first = groupby.start_iend.first()
        last = groupby.start_iend.last()
        index = pd.concat([first, last])
        data = pd.concat([last, first])
        result = data.set_axis(index)
        assert not result.index.has_duplicates
        assert not result.duplicated().any()
        self[key] = result
        result = self[key]
        return result

    @property
    def stop_other_iend(self) -> pd.Series:
        """Other iend of the node at self.stop_iend"""
        key = 'stop_other_iend'
        if key in self:
            return self[key]
        result = (
            self.start_other_iend
            .reindex(self.stop_iend)
            .values
        )

        self[key] = result
        result = self[key]
        return result

    @property
    def stop_degree(self) -> pd.Series:
        key = 'stop_degree'
        if key in self:
            return self[key]
        result = (
            self.lines.nodes.degree
            .loc[self.stop_inode]
            .values
        )
        self[key] = result
        result = self[key]
        return result

    @property
    def iunion(self) -> pd.Series:
        key = 'iunion'
        if key in self:
            return self[key]
        polygons = self.lines.pednet.union
        geometry = self.geometry
        iloc, idissolved = polygons.sindex.nearest(geometry, return_distance=False)
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
        nodes = self.lines.nodes
        x1 = nodes.threshold.loc[self.start_inode].values
        x2 = nodes.threshold.loc[self.stop_inode].values
        result = np.maximum(x1, x2)
        self[key] = result
        result = self[key]
        return result

    @property
    def start_tuple(self) -> pd.Series:
        key = 'start_tuple'
        if key in self:
            return self[key]
        result = (
            self.nodes.tuple
            .loc[self.start_inode]
            .values
        )
        self[key] = result
        result = self[key]
        return result

    @property
    def stop_tuple(self) -> pd.Series:
        key = 'stop_tuple'
        if key in self:
            return self[key]
        result = (
            self.nodes.tuple
            .loc[self.stop_inode]
            .values
        )
        self[key] = result
        result = self[key]
        return result

    @property
    def feature(self) -> pd.Series:
        if 'feature' in self:
            return self['feature']
        feature = self.lines.feature.loc[self.iline].values
        self['feature'] = feature
        result = self['feature']
        return result

    def __set_name__(self, owner, name):
        self.__name__ = name
