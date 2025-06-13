from __future__ import annotations
import geopandas as gpd
from pandas.core.dtypes.inference import is_named_tuple

from sklearn.neighbors import NearestNeighbors

from functools import *
from typing import *

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.ops
from geopandas import GeoDataFrame
from pandas import Series
from ..fixed import GeoDataFrameFixed

if False:
    from .pednet import PedNet
    import folium

"""
without magicpandas,
input: lines
output: nodes, aggregated

one to drop degree=2 noeds
one to extract node information

"""


@singledispatch
def explore(
        self,
        name=None,
        geometry=None,
        *args,
        **kwargs
) -> folium.Map:
    """Convenience wrapper to GeoDataFrame.explore"""
    import folium
    self = self.copy()
    _ = self.geometry
    if isinstance(self, pd.Series):
        # noinspection PyTypeChecker
        self = (
            self
            .reset_index()
            .rename(columns={'geometry': 'geometry'})
        )
        geometry = 'geometry'
    kwargs['tiles'] = kwargs.setdefault('tiles', 'cartodbdark_matter')
    style_kwargs = kwargs.setdefault('style_kwds', {})
    style_kwargs.setdefault('weight', 5)
    style_kwargs.setdefault('radius', 5)

    if (
            not geometry
            and isinstance(self, GeoDataFrame)
    ):
        geometry = self._geometry_column_name
    if name is None:
        name = geometry

    if not isinstance(self, gpd.GeoSeries):
        _ = self[geometry]

        loc = self.dtypes != 'geometry'
        loc &= self.dtypes != 'object'
        loc |= self.columns == geometry
        # todo: better way other than checking first row?
        try:
            is_string = np.fromiter((
                isinstance(x, str)
                for x in self.iloc[0]
            ), dtype=bool, count=len(self.columns))
            loc |= is_string
        except IndexError:
            ...

        columns = self.columns[loc]
        loc = self[geometry].notna().values
        max_zoom = kwargs.setdefault('max_zoom', 28)
        zoom_start = kwargs.setdefault('zoom_start', 14)
        if not loc.any():
            try:
                return kwargs['m']
            except KeyError:
                result = folium.Map(name=name, max_zoom=max_zoom, zoom_start=zoom_start)
            return result
        self = (
            self.loc[loc, columns]
            .reset_index()  # reset index so it shows in folium
            .set_geometry(geometry)
            .pipe(GeoDataFrame)
        )
    else:
        if not self.name:
            self.name = name
        self = (
            self
            .reset_index()
        )

    if 'MultiPoint' in self.geom_type.unique():
        # LayerControl doesn't work with MultiPoints
        self = self.explode(column=geometry, ignore_index=True)

    m = GeoDataFrame.explore(
        self,
        *args,
        **kwargs,
        name=name,
    )
    return m


class Nodes(
    GeoDataFrameFixed
):
    x: Series
    y: Series
    degree: Series
    lines: Lines

    def __get__(self, instance: Lines, owner) -> Self:
        if instance is None:
            raise ValueError
        if 'nodes' not in instance.attrs:
            instance._nodes()

        result = instance.attrs['nodes']
        result.lines = instance
        return result

    @property
    def inode(self) -> pd.Index:
        return self.index

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
            ['start_end']
            .apply(tuple)
            .loc[self.inode]
            .values
        )
        self[key] = result
        result = self[key]
        return result


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
        cols = 'iline geometry start_inode stop_inode start_end stop_end'.split()
        lines = (
            instance
            .reset_index()
            [cols]
        )
        reverse = (
            instance
            .set_geometry(instance.reverse())
            .assign(
                stop_node=instance.start_inode.values,
                start_inode=instance.stop_node.values,
                stop_end=instance.start_end.values,
            )
            .set_axis(instance.start_end)
        )
        concat = lines, reverse
        result = (
            pd.concat(concat)
            .pipe(Edges)
        )
        instance.__dict__[self.__name__] = result
    return result


class Edges(
    GeoDataFrameFixed,
):
    stop_end: pd.Series
    lines: Lines = None
    locals().update(
        __get__=__get__,
    )

    def __set_name__(self, owner, name):
        self.__name__ = name

    @property
    def start_icoord(self) -> pd.Index:
        return self.index

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
    def start_inode(self):
        ...

    @property
    def stop_node(self):
        ...

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
        result = (
                self.lines.pednet.union
                .reindex(self.iunion)
                .distance(self.geometry, align=False)
                .values
                * (2 ** 0.5)
        )
        self[key] = result
        result = self[key]
        return result

    @property
    def tuple(self):
        ...


class Lines(
    gpd.GeoDataFrame
):
    start_end: pd.Series
    stop_end: pd.Series
    pednet: PedNet = None

    @Edges
    def edges(self):
        ...

    @property
    def iunion(self):
        if self.pednet is None:
            raise ValueError('PedNet not set')

    def _nodes(self) -> Series:
        lines = shapely.get_parts(self.geometry)
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
        crs = self.crs
        data = dict(degree=degree, x=x, y=y)
        nodes = Nodes(data, geometry=geometry, crs=crs)
        nodes.index.name = 'inode'
        self.attrs['nodes'] = nodes

        inode = (
            nodes
            .reset_index()
            .set_index('x y'.split())
            .inode
            .loc[haystack]
        )
        start_inode = inode.iloc[::2].values
        stop_inode = inode.iloc[1::2].values
        self['start_inode'] = start_inode
        self['stop_inode'] = stop_inode

    @property
    def start_inode(self) -> pd.Series:
        if 'start_inode' not in self:
            self._nodes()
        return self['start_inode']

    @property
    def stop_inode(self) -> pd.Series:
        if 'stop_inode' not in self:
            self._nodes()
        return self['stop_inode']

    # @cached_property
    # def start_inode(self) -> Nodes:
    #     return self.nodes.loc[self.start_inode]
    #
    # @cached_property
    # def stop_node(self) -> Nodes:
    #     return self.nodes.loc[self.start_inode]

    @property
    def nodes(self) -> Nodes:
        if 'nodes' not in self.attrs:
            self._nodes()
        return self.attrs['nodes']

    @classmethod
    def from_frame(
            cls,
            frame: gpd.GeoDataFrame
    ) -> Self:
        geometry = frame.geometry
        loc = geometry.geom_type != 'LineString'
        loc &= geometry.geom_type != 'MultiLineString'
        if np.any(loc):
            count = np.sum(loc)
            msg = f'{count} geometries are not LineString or MultiLineString'
            raise ValueError(msg)

        loc = geometry.geom_type == 'MultiLineString'
        crs = frame.crs
        concat = [shapely.get_parts(geometry.loc[loc])]
        loc = geometry.geom_type == 'LineString'
        concat.append(geometry.loc[loc])
        geometry = np.concatenate(concat)
        stop = len(geometry) * 2
        index = pd.RangeIndex(0, stop, 2, name='start_end')
        stop_end = np.arange(1, len(geometry) * 2, step=2)
        data = dict(
            stop_end=stop_end,
        )
        result = cls(
            geometry=geometry,
            data=data,
            crs=crs,
            index=index,
        )

        return result

    @cached_property
    def drop2nodes(self, ) -> Self:
        union = self.geometry.unary_union
        multiline = shapely.ops.linemerge(union)
        lines = shapely.get_parts(multiline)
        result = self.__class__(
            geometry=lines,
            crs=self.crs,
        )
        return result

    def explore(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            line='grey',
            node='red',
            **kwargs,
    ) -> folium.Map:
        import folium
        m = explore(
            self.geometry,
            color=line,
            name='lines',
            *args,
            **kwargs,
            tiles=tiles,
            m=m,
        )
        nodes = self.nodes
        loc = nodes.index.isin(self.start_inode.values)
        loc |= nodes.index.isin(self.stop_inode.values)
        nodes = nodes.loc[loc]
        m = explore(
            nodes,
            color=node,
            name='node',
            *args,
            **kwargs,
            tiles=tiles,
            m=m,
        )
        folium.LayerControl().add_to(m)
        return m


if __name__ == '__main__':
    import geopandas as gpd

    file = '/home/arstneio/PycharmProjects/kashi/src/tile2net/artifacts/static/brooklyn.feather'
    result = (
        gpd.read_feather(file)
        .pipe(Lines.from_frame)
        .drop2nodes
    )
    print(f'{len(result)=}')
    print(f'{len(result)=}')
    result.explore()
