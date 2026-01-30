from __future__ import annotations

from typing import *

import geopandas as gpd

from . import mintrees, stubs
from ..frame.framewrapper import FrameWrapper

_ = mintrees, stubs

if TYPE_CHECKING:
    from .pednet import PedNet


class Curbs(
    FrameWrapper
):
    instance: PedNet = None
    __name__ = 'curbs'

    def __get__(
            self,
            instance: PedNet,
            owner: type[PedNet]
    ) -> Curbs:
        self: Self = FrameWrapper._get(self, instance, owner)
        cache = instance.__dict__
        key = self.__name__
        if instance is None:
            return self
        if key in cache:
            result = cache[key]
            if result.instance is not instance:
                raise NotImplementedError

        else:
            loc = instance.feature == 'road'

            road: PedNet = instance.loc[loc]
            not_road: PedNet = instance.loc[~loc]
            boundary: gpd.GeoSeries = road.geometry.boundary
            union = not_road.geometry.union_all()
            difference: gpd.GeoSeries = boundary.difference(union)
            difference.explode()



            cache[key] = result

        result.instance = instance
        return result

