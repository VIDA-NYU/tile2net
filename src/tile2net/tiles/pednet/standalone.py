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

if False:
    from .pednet import PedNet
    import folium
    from .stubs import Stubs
    from .mintrees import Mintrees

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
            ['start_end']
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
        _ = (
            instance.start_inode, instance.stop_inode,
            instance.start_end, instance.stop_end,
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
                stop_end=instance.start_end.values,
                start_end=instance.stop_end.values,
                start_x=instance.stop_x.values,
                start_y=instance.stop_y.values,
                stop_x=instance.start_x.values,
                stop_y=instance.start_y.values,
            )
        )
        concat = lines, reverse
        result = (
            pd.concat(concat, )
            .set_index('start_end')
            .pipe(self.__class__)
        )
        instance.__dict__[self.__name__] = result

    result.lines = instance

    return result


class Edges(
    GeoDataFrameFixed,
):
    stop_end: pd.Series
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
    def start_end(self) -> pd.Index:
        # return self.index
        key = 'start_end'
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

    def __set_name__(self, owner, name):
        self.__name__ = name

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
    def start_tuple(self) -> pd.Series:
        key = 'start_tuple'
        if key in self:
            return self[key]
        result = self.nodes.tuple.loc[self.start_inode]
        self[key] = result
        result = self[key]
        return result

    @property
    def stop_tuple(self) -> pd.Series:
        key = 'stop_tuple'
        if key in self:
            return self[key]
        result = self.nodes.tuple.loc[self.stop_inode]
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


class Lines(
    gpd.GeoDataFrame
):
    start_end: pd.Series
    stop_end: pd.Series
    pednet: PedNet = None
    stubs: Stubs
    mintrees: Mintrees
    start_x: pd.Series
    start_y: pd.Series
    stop_x: pd.Series
    stop_y: pd.Series
    stop_end: pd.Series
    start_end: pd.Series

    __keep__ = 'geometry start_x start_y stop_x stop_y start_end stop_end'.split()

    @Edges
    def edges(self):
        ...

    @Nodes
    def nodes(self):
        ...

    @property
    def iunion(self):
        if self.pednet is None:
            raise ValueError('PedNet not set')

    @property
    def start_inode(self) -> pd.Series:
        if 'start_inode' not in self:
            cols = 'start_x start_y'.split()
            loc = pd.MultiIndex.from_frame(self[cols])
            result = (
                self.nodes
                .reset_index()
                .set_index('x y'.split())
                ['inode']
                .loc[loc]
                .values
            )
            self['start_inode'] = result
            assert self.start_inode.isin(self.nodes.inode.values).all()

        return self['start_inode']

    @property
    def stop_inode(self) -> pd.Series:
        if 'stop_inode' not in self:
            cols = 'stop_x stop_y'.split()
            loc = pd.MultiIndex.from_frame(self[cols])
            result = (
                self.nodes
                .reset_index()
                .set_index('x y'.split())
                ['inode']
                .loc[loc]
                .values
            )
            self['stop_inode'] = result
            assert self.stop_inode.isin(self.nodes.inode.values).all()

        return self['stop_inode']

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
        start_end = np.arange(0, stop, step=2)
        stop_end = np.arange(1, len(geometry) * 2, step=2)

        lines = shapely.get_parts(frame.geometry)
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
        iloc = ilast

        start_x = coords[ifirst, 0]
        start_y = coords[ifirst, 1]
        stop_x = coords[ilast, 0]
        stop_y = coords[ilast, 1]

        data = dict(
            stop_end=stop_end,
            start_end=start_end,
            start_x=start_x,
            start_y=start_y,
            stop_x=stop_x,
            stop_y=stop_y,
        )
        result = cls(geometry=geometry, data=data, crs=crs, )
        result.index.name = 'iline'

        return result

    # @cached_property
    # def drop2nodes(self) -> Self:
    #     union = self.geometry.unary_union
    #     multiline = shapely.ops.linemerge(union)
    #     lines = shapely.get_parts(multiline)
    #     result = self.__class__(
    #         geometry=lines,
    #         crs=self.crs,
    #     )
    #     return result

    def drop2nodes(self):
        ...


    def visualize(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            line_color='grey',
            node_color='red',
            polygon_color='grey',
            simplify=None,
            **kwargs,
    ) -> folium.Map:
        import folium
        if polygon_color:
            m = explore(
                self.pednet.union,
                *args,
                color=polygon_color,
                name=f'polygons',
                tiles=tiles,
                simplify=simplify,
                m=m,
                style_kwds=dict(
                    dashArray='5, 15',
                    fill=False,
                ),
                **kwargs,
            )

        m = explore(
            self.geometry,
            color=line_color,
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
            color=node_color,
            name='nodes',
            *args,
            **kwargs,
            tiles=tiles,
            m=m,
        )
        folium.LayerControl().add_to(m)
        return m

    @property
    def iline(self) -> pd.Index:
        key = 'iline'
        if key in self.index.names:
            return self.index.get_level_values(key)
        else:
            return self[key]


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
