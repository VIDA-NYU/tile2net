from __future__ import annotations
from ..benchmark import benchmark
from ..explore import explore
from ..vectiles.mask2poly import Mask2Poly
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

    # elif os.path.exists(instance.outdir.lines.file):
    #     file = instance.outdir.lines.file
    #     msg = f"Reading {self.__name__} from {file}"
    #     logger.info(msg)
    #     result = (
    #         gpd.read_parquet(file)
    #         .pipe(self.__class__)
    #     )
    #     result.intiles = instance
    #     instance.__dict__[self.__name__] = result

    else:
        # msg = f'Vectorizing polygons for {instance}'
        # logger.debug(msg)
        result = (
            instance.vectiles.polygons
            .stack(future_stack=True, )
            .to_frame(name='geometry')
            .dissolve(level=2)
            .explode()
            .rename_axis('feature')
            .pipe(Mask2Poly)
            .postprocess()
            .pipe(Polygons)
        )

        instance.__dict__[self.__name__] = result
        # file = instance.outdir.polygons.file
        # msg = f"Writing {instance.__name__}.{self.__name__} to {file}"
        # logger.info(msg)

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

    def explore(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            dash='5, 20',
            simplify=None,
            **kwargs,
    ) -> folium.Map:

        import folium
        feature2color = cfg.label2color
        it = self.groupby('feature', observed=False)
        for feature, frame in it:
            color = feature2color[feature]
            m = explore(
                frame,
                *args,
                color=color,
                name=feature,
                tiles=tiles,
                simplify=simplify,
                style_kwds=dict(
                    fill=False,
                    dashArray=dash,
                ),
                m=m,
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m
