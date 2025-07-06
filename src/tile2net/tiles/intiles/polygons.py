from __future__ import annotations
from ..explore import explore
from tile2net.tiles.cfg.logger import logger
from ..cfg import cfg

from typing import *

import pandas as pd
from geopandas import GeoSeries
from shapely import *

from ..fixed import GeoDataFrameFixed
import os
import geopandas as gpd

if False:
    from .intiles import InTiles
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

    elif os.path.exists(instance.outdir.lines.file):
        result = (
            gpd.read_parquet(instance.outdir.lines.file)
            .pipe(self.__class__)
        )
        result.intiles = instance
        instance.__dict__[self.__name__] = result

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

    def unlink(self):
        """Delete the polygons file."""
        file = self.intiles.outdir.polygons.file
        if os.path.exists(file):
            os.remove(file)
        self.intiles.__dict__.pop(self.__name__, None)
