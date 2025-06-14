from __future__ import annotations
from ..cfg import cfg
from ..explore import explore
from tile2net.logger import logger

from typing import *

import geopandas as gpd
import numpy as np
import shapely
from centerline.geometry import Centerline

from ..fixed import GeoDataFrameFixed

if False:
    import folium
    from .pednet import PedNet


def __get__(
        self,
        instance: PedNet,
        owner: type[PedNet]
) -> Self:
    self.instance = instance
    self.owner = owner
    return self


def __get__(
        self: Union,
        instance: PedNet,
        owner: type[PedNet]
) -> Union:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        msg = 'Computing the geometric union of all pedestrian features.'
        logger.debug(msg)
        loc = ~instance.feature.isin(cfg.polygon.borders)
        geometry = (
            instance
            .loc[loc]
            .geometry
            .union_all()
        )
        data = shapely.get_parts(geometry)
        crs = instance.crs
        geometry = gpd.GeoSeries(data, crs=crs)
        result = Union(geometry=geometry)
        result.index.name = 'iunion'
        instance.__dict__[self.__name__] = result
        return result

    result.instance = instance
    return result


class Union(
    GeoDataFrameFixed
):
    locals().update(
        __get__=__get__,
    )
    instance: PedNet = None
    __name__ = 'union'

    def explore(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            line='grey',
            node='red',
            simplify: float = None,
            dash='5, 20',
            **kwargs,
    ) -> folium.Map:
        import folium
        features = self.instance.features
        feature2color = features.color.to_dict()
        it = features.groupby(level='feature', observed=False)

        for feature, frame in it:
            color = feature2color[feature]
            m = explore(
                frame,
                geometry='mutex',
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
