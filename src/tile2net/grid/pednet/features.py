from __future__ import annotations
from ..frame.framewrapper import FrameWrapper
from .. import frame
from ..explore import explore
from tile2net.grid.cfg.logger import logger
from ..cfg import cfg
from ...grid.frame.namespace import namespace

from typing import *

import pandas as pd
from geopandas import GeoSeries
from shapely import *

if False:
    import folium
    from .pednet import PedNet


class Features(
    FrameWrapper
):
    __name__ = 'features'
    _pednet: PedNet = None

    def _get(
            self,
            instance: PedNet,
            owner: type[PedNet]
    ) -> Self:
        self: Self = namespace._get(self, instance, owner)
        cache = instance.frame.__dict__
        key = self.__name__
        if instance is None:
            return self
        if key in cache:
            result: Self = cache[key]
            if instance is not result.instance:
                del cache[key]
                return self._get(instance, owner)
        else:
            result = (
                instance
                .frame
                .dissolve(by='feature')
                .pipe(self.from_frame, wrapper=self)
            )
            cache[key] = result
        result._pednet = instance

        return result

    locals().update(
        __get__=_get
    )

    def __set__(
            self,
            instance: PedNet,
            value: type[PedNet],
    ):
        value.__name__ = self.__name__
        instance.__dict__[self.__name__] = value

    @frame.column
    def above(self):
        """Unary union of all features above the given feature."""
        # ordered = self.sort_values(self.z_order.name, ascending=False)
        ordered = (
            self.frame
            .sort_values(self.z_order.name, ascending=False)
        )
        geoms = ordered.geometry.values
        out = []
        cum = GeometryCollection()
        for g in geoms:
            out.append(cum)
            cum = cum.union(g)
        result = GeoSeries(out, index=ordered.index, crs=self.crs)
        return result

    @frame.column
    def mutex(self) -> GeoSeries[GeometryCollection]:
        """Mutually exclusive geometries based on z_order"""
        msg = f'Computing mutex for {self.__name__}'
        logger.debug(msg)
        result = self.geometry.difference(self.above)
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

    @frame.column
    def other(self) -> GeoSeries[GeometryCollection]:
        """Geometry union of every feature excluding the feature itself."""
        data = (
            self._pednet
            .frame
            .union_all()
            .difference(self.geometry)
        )
        result = GeoSeries(data, self.index, self.crs)
        return result

    @frame.column
    def feature(self) -> pd.Index:
        """Feature index values."""
        return self.index.get_level_values('feature')

    @frame.column
    def color(self) -> pd.Series:
        """Color mapping for features."""
        result = pd.Series(cfg.label2color, dtype='string')
        return result

    @frame.column
    def z_order(self) -> pd.Series:
        """Z-order of the features."""
        result = pd.Series(cfg.polygon.z_order)
        return result

    def explore(
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
            it = features.frame.groupby(
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

        it = features.frame.groupby(level='feature', observed=False)
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
