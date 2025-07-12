from __future__ import annotations
import numpy as np
import shapely

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

    else:

        vectiles = instance.vectiles
        n = len(instance.vectiles.polygons)
        grid_size = max(
            (affine.a ** 2 + affine.e ** 2) ** .5
            for affine in vectiles.affine_params
        )
        # factor = 1e-1
        # grid_size = min(
        #     max(abs(affine.a), abs(affine.e))
        #     for affine in vectiles.affine_params
        # ) * factor
        n_polygons = (
            vectiles.polygons
            .apply(shapely.get_num_geometries)
            .to_numpy()
            .sum()
        )

        n_features = len(vectiles.polygons.columns)
        msg = (
            f'Aggregating {n_polygons} polygons from {n} tiles and '
            f'{n_features} features into a single vector with grid '
            f'size {grid_size:.2e}. This may take a while.'
        )

        with benchmark(msg, level='info'):
            result = (
                instance.vectiles.polygons
                .stack(future_stack=True)
                .set_precision(grid_size=grid_size)
                .to_frame(name='geometry')
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
