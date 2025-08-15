from __future__ import annotations
from ..frame.framewrapper import FrameWrapper
from ..benchmark import benchmark
from ..cfg import cfg
from ..explore import explore

from typing import *
from typing import Self

import geopandas as gpd
import shapely

from ...grid.frame.namespace import namespace

if False:
    import folium
    from .pednet import PedNet


class Union(

    FrameWrapper
):
    """
    Union of all pedestrian features
    """
    instance: PedNet = None
    __name__ = 'union'

    def _get(
            self,
            instance: PedNet,
            owner: type[PedNet]
    ) -> Union:
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
            result: Self = (
                gpd.GeoSeries(data, crs=crs)
                .to_frame(name='geometry')
                .pipe(instance.from_frame, wrapper=self)
            )

            result.index.name = 'iunion'
            cache[key] = result
            # result.collection = collection
            return result

        result.instance = instance
        return result

    locals().update(
        __get__=_get,
    )

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
        it = features.frame.groupby(level='feature', observed=False)

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
