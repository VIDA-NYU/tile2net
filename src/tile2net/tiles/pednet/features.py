from __future__ import annotations
from ..explore import explore
from tile2net.logger import logger
from ..cfg import cfg

import itertools
from typing import *

import geopandas as gpd
import pandas as pd
from geopandas import GeoSeries
from shapely import *

from ..fixed import GeoDataFrameFixed

if False:
    import folium
    from .pednet import PedNet


def __get__(
        self,
        instance: PedNet,
        owner: type[PedNet]
) -> Self:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        result = (
            instance
            .dissolve(by='feature')
            .pipe(Features)
        )
        instance.__dict__[self.__name__] = result
    result._pednet = instance

    return result


class Features(
    GeoDataFrameFixed
):
    __name__ = 'features'
    _pednet: PedNet = None
    locals().update(
        __get__=__get__,
    )

    def __set__(
            self,
            instance: PedNet,
            value: type[PedNet],
    ):
        value.__name__ = self.__name__
        instance.__dict__[self.__name__] = value

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
        key = 'mutex'
        if key in self:
            return self[key]
        msg = f'Computing {key} for {self.__name__}'
        logger.debug(msg)
        result = self.geometry.difference(self.above)
        self[key] = result
        result = self[key]
        return result

    @property
    def original(self) -> GeoSeries[GeometryCollection]:
        """Original geometries of the features."""
        key = 'original'
        try:
            return self[key]
        except KeyError as e:
            msg = (
                f'features.original has not been set; it should have '
                f'been set during PedNet instantiation'
            )
            raise AttributeError(msg) from e

    @original.setter
    def original(self, value: GeoSeries[GeometryCollection]):
        """Set the original geometries of the features."""
        if not isinstance(value, GeoSeries):
            raise TypeError('original must be a GeoSeries')
        self['original'] = value

    @property
    def other(self) -> GeoSeries[GeometryCollection]:
        """Geometry union of every feature excluding the feature itself."""
        key = 'other'
        if key in self:
            return self[key]

        # logger.debug('Computing %s for %s', key, self.__name__)
        msg = f'Computing the u'
        data = (
            self._pednet
            .union_all()
            .difference(self.geometry)
        )
        result = GeoSeries(data, self.index, self.crs)

        self[key] = result
        return self[key]

    @property
    def feature(self) -> pd.Index:
        return self.index.get_level_values('feature')

    @property
    def color(self) -> pd.Series:
        key = f'color'
        if key in self:
            return self[key]
        self[key] = pd.Series(cfg.label2color, dtype='string')
        result = self[key]
        return result

    @property
    def z_order(self) -> pd.Series:
        """Z-order of the features."""
        key = f'z_order'
        if key in self:
            return self[key]
        result = pd.Series(cfg.polygon.z_order)
        self[key] = result
        result = self[key]
        return result

    def visualize(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            simplify: float = None,
            dash='5, 20',
            attr: str = None,
            **kwargs,
    ) -> folium.Map:
        import folium
        features = self
        feature2color = features.color.to_dict()
        _ = features.mutex

        if 'original' in features:
            it = features.groupby(
                level='feature',
                observed=False,
            )
            for feature, frame in it:
                color = feature2color[feature]
                m = explore(
                    frame,
                    geometry='original',
                    *args,
                    color=color,
                    name=f'{feature} (original)',
                    tiles=tiles,
                    simplify=simplify,
                    m=m,
                    style_kwds=dict(
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.05,
                        weight=0,  # no stroke
                    ),
                    highlight=False,
                    **kwargs,
                )

        it = features.groupby(level='feature', observed=False)
        if attr:
            getattr(self, attr)
            geometry = attr
        else:
            geometry = 'mutex'

        for feature, frame in it:
            color = feature2color[feature]
            m = explore(
                frame,
                geometry=geometry,
                *args,
                color=color,
                name=f'{feature} polygons',
                tiles=tiles,
                simplify=simplify,
                m=m,
                style_kwds=dict(
                    fill=False,
                    dashArray=dash,
                ),
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m
