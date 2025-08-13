from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd

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



class Edges(
    FrameWrapper,
):
    stop_iend: pd.Series
    iline: pd.Series
    start_x: pd.Series
    start_y: pd.Series
    stop_x: pd.Series
    stop_y: pd.Series
    start_inode: pd.Series
    stop_inode: pd.Series


    def _get(
            self,
            instance: Lines,
            owner
    ) -> Edges:
        self: Self = namespace._get(self, instance, owner)
        cache = instance.frame.__dict__
        key = self.__name__
        if instance is None:
            return self
        if key in cache:
            result = instance.__dict__[self.__name__]
        else:
            cols = 'iline geometry start_inode stop_inode start_iend stop_iend'.split()
            _ = (
                instance.start_inode, instance.stop_inode,
                instance.start_iend, instance.stop_iend,
            )
            lines = (
                instance.frame
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
                .pipe(self.from_frame, wrapper=self)
            )
            cache[key] = result

        result.lines = instance

        return result

    locals().update(
        __get__=_get,
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

    @frame.column
    def nodes(self):
        return self.lines.nodes

    @frame.column
    def start_iend(self) -> pd.Index:
        key = 'start_iend'
        if key in self.index.names:
            return self.index.get_level_values(key)
        else:
            return self.frame[key]


    @frame.column
    def start_degree(self) -> pd.Series:
        result = (
            self.lines.nodes.degree
            .loc[self.start_inode]
            .values
        )
        return result


    @frame.column
    def start_shared_iend(self) -> pd.Series:
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

        result = other.where(loc)
        return result

    @frame.column
    def stop_shared_iend(self) -> pd.Series:
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

        result = other.where(loc)
        return result

    @frame.column
    def start_other_iend(self) -> pd.Series:
        """Other iend of the node at self.start_iend"""
        loc = self.start_degree == 2
        edges: Edges = self.loc[loc]
        groupby = (
            edges
            .reset_index()
            .groupby(edges.start_inode.values, sort=False)
        )
        if not len(groupby):
            result = pd.Series(dtype='UInt32')
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
        return result

    @frame.column
    def stop_other_iend(self) -> pd.Series:
        """Other iend of the node at self.stop_iend"""
        result = (
            self.start_other_iend
            .reindex(self.stop_iend)
            .values
        )
        return result

    @frame.column
    def stop_degree(self) -> pd.Series:
        result = (
            self.lines.nodes.degree
            .loc[self.stop_inode]
            .values
        )
        return result

    @frame.column
    def iunion(self) -> pd.Series:
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
        return result

    @frame.column
    def threshold(self) -> pd.Series:
        nodes = self.lines.nodes
        x1 = nodes.threshold.loc[self.start_inode].values
        x2 = nodes.threshold.loc[self.stop_inode].values
        result = np.maximum(x1, x2)
        return result

    @frame.column
    def start_tuple(self) -> pd.Series:
        result = (
            self.nodes.tuple
            .loc[self.start_inode]
            .values
        )
        return result

    @frame.column
    def stop_tuple(self) -> pd.Series:
        result = (
            self.nodes.tuple
            .loc[self.stop_inode]
            .values
        )
        return result

    @frame.column
    def feature(self) -> pd.Series:
        feature = self.lines.feature.loc[self.iline].values
        return feature

    def __set_name__(self, owner, name):
        self.__name__ = name
