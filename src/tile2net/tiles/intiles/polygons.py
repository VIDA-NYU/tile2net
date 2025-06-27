from __future__ import annotations
from ..explore import explore
from tile2net.tiles.cfg.logger import logger
from ..cfg import cfg

from typing import *

import pandas as pd
from geopandas import GeoSeries
from shapely import *

from ..fixed import GeoDataFrameFixed

if False:
    from .intiles import  InTiles
    import folium

def __get__(
        self: Polygons,
        instance: InTiles,
        owner: type[InTiles]
) -> Polygons:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        result = (
            instance
            .dissolve(by='feature')
            .explode()
            .pipe(Polygons)
        )
        instance.__dict__[self.__name__] = result

    return result


class Polygons(
    GeoDataFrameFixed
):
    __name__ = 'polygons'
    locals().update(
        __get__=__get__,
    )
