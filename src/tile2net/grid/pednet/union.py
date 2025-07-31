from __future__ import annotations
from ..benchmark import benchmark
from ..cfg import cfg
from ..explore import explore

from typing import *

import geopandas as gpd
import shapely

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
        loc = ~instance.feature.isin(cfg.polygon.borders)
        msg = f'Computing the geometric union of all {loc.sum()} pedestrian features.'
        # logger.debug(msg)
        with benchmark(msg):
            collection = (
                instance
                .loc[loc]
                .geometry
                .union_all()
            )
        data = shapely.get_parts(collection)
        crs = instance.crs
        geometry = gpd.GeoSeries(data, crs=crs)
        result = Union(geometry=geometry)
        result.index.name = 'iunion'
        instance.__dict__[self.__name__] = result
        # result.collection = collection
        return result

    result.instance = instance
    return result


class Union(
    GeoDataFrameFixed
):
    """
    Union of all pedestrian features
    """
    locals().update(
        __get__=__get__,
    )
    instance: PedNet = None
    __name__ = 'union'


    def explore(
            self,
            *args,
            grid='cartodbdark_matter',
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
                grid=grid,
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
