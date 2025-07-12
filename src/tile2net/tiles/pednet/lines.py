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
from .edges import Edges
from .nodes import Nodes

if False:
    from .pednet import PedNet
    import folium
    from .stubs import Stubs
    from .mintrees import Mintrees

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



class Lines(
    gpd.GeoDataFrame
):
    start_iend: pd.Series
    stop_iend: pd.Series
    pednet: PedNet = None
    stubs: Stubs
    mintrees: Mintrees
    start_x: pd.Series
    start_y: pd.Series
    stop_x: pd.Series
    stop_y: pd.Series
    stop_iend: pd.Series
    start_iend: pd.Series

    __keep__ = 'geometry start_x start_y stop_x stop_y start_iend stop_iend'.split()

    @Edges
    def edges(self):
        ...

    @Nodes
    def nodes(self):
        ...

    @property
    def feature(self) -> pd.Series:
        try:
            return self['feature']
        except KeyError as e:
            raise KeyError(
                'Feature not set. Please set feature column '
                'before accessing Lines.feature'
            ) from e

    @feature.setter
    def feature(self, value: pd.Series):
        self['feature'] = value

    @feature.deleter
    def feature(self):
        del self['feature']

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

        result = (
            frame
            .reset_index()
            .explode()
        )

        stop = len(geometry) * 2
        start_iend = np.arange(0, stop, step=2)
        stop_iend = np.arange(1, len(geometry) * 2, step=2)

        lines = shapely.get_parts(result.geometry)
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

        assign = dict(
            stop_iend=stop_iend,
            start_iend=start_iend,
            start_x=start_x,
            start_y=start_y,
            stop_x=stop_x,
            stop_y=stop_y,
        )
        result = (
            result
            .assign(**assign)
            .pipe(cls)
        )

        result.index.name = 'iline'

        return result

    def drop2nodes(self):
        visited = set()
        edges = self.edges

        edges.start_other_iend

        iloc = (
            edges.start_shared_iend
            .notna()
            .argsort(kind='mergesort')
        )
        edges = edges.iloc[iloc]

        iend2inext = (
            edges.stop_shared_iend
            .dropna()
            .to_dict()
        )

        iend2iline = edges.iline.to_dict()
        it = zip(edges.start_iend.values, edges.iline.values)
        iline = -1

        list_iend = []
        list_iline = []
        for (start_iend, start_iline) in it:
            if start_iline in visited:
                continue
            iline += 1
            visited.add(start_iline)
            list_iend.append(start_iend)
            list_iline.append(iline)

            while start_iend in iend2inext:
                next_iend = iend2inext[start_iend]
                next_iline = iend2iline[next_iend]
                if next_iline in visited:
                    break
                visited.add(next_iline)
                list_iend.append(next_iend)
                list_iline.append(iline)
                start_iend = next_iend

        data = {}
        iend = np.array(list_iend)
        geometry: gpd.GeoSeries = self.edges.geometry.loc[iend]
        coords = shapely.get_coordinates(geometry, include_z=False)
        repeat = shapely.get_num_points(geometry)
        iline = np.array(list_iline)
        try:
            feature = (
                self.edges.feature
                .loc[iend]
                .groupby(iline)
                .first()
                .values
            )
        except KeyError:
            ...
        else:
            data['feature'] = feature
        iline = iline.repeat(repeat)
        lines = shapely.linestrings(coords, indices=iline)
        lines = shapely.remove_repeated_points(lines)
        result = self.__class__(
            geometry=lines,
            crs=self.crs,
            data=data,
        )
        return result

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
