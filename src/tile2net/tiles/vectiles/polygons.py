from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
)
from multiprocessing import cpu_count
from typing import *

import geopandas as gpd
import pandas as pd
import shapely
from geopandas.array import GeometryDtype

from ..benchmark import benchmark
from ..cfg import cfg
from ..cfg.logger import logger
from ..explore import explore
from ..fixed import GeoDataFrameFixed

if False:
    from .vectiles import VecTiles
    import folium


def __get__(
        self: Polygons,
        instance: VecTiles,
        owner: type[VecTiles]
) -> Polygons:
    if instance is None:
        result = self

    elif self.__name__ not in instance.__dict__:

        msg = f'Loading {instance.__name__}.polygons'
        logger.debug(msg)

        idx_names = list(instance.index.names)

        def _read(idx_path):
            idx, path = idx_path
            gdf = (
                gpd
                .read_parquet(path)
                .reset_index(names='feature')
            )
            if not isinstance(idx, tuple):
                idx = (idx,)
            for name, val in zip(idx_names, idx):
                gdf[name] = val
            gdf['src'] = 'polygons'
            return gdf

        tasks = [
            (i, p)
            for i, p
            in instance.file.polygons.items()
        ]

        with ThreadPoolExecutor() as ex:
            frames = list(ex.map(_read, tasks))

        msg = f'Dissolving by feature and tile'
        nparts = cpu_count() * 2
        # Client(processes=True)

        # multi2single
        # frames[0].geom_type.unique()
        # features = frames[0].feature.unique()
        # instance.index.repeat(len(features))

        result = (
            pd.concat(frames, copy=False)
            .pipe(gpd.GeoDataFrame)
            # .explode()
        )
        import numpy as np
        # np.sum(result.geom_type == 'MultiPolygon')
        loc = result.geom_type == 'MultiPolygon'
        result.loc[loc]


        cols = 'xtile ytile feature'.split()
        # index = (
        #     result[cols]
        #     .pipe(pd.MultiIndex.from_frame)
        #     .drop_duplicates()
        # )
        # multi2single = pd.Series(np.arange(len(index)), index=index)
        # single2multi = multi2single.index
        # loc = pd.MultiIndex.from_frame(result[cols])
        # single = multi2single.loc[loc]

        with benchmark(msg):
            # result = (
            #     # pd.concat(frames, copy=False)
            #     # .pipe(gpd.GeoDataFrame)
            #     # .explode()
            #     result
            #     .assign(single=single.values)
            #     .pipe(dg.from_geopandas, npartitions=nparts)
            #     .persist()
            #     .dissolve(
            #         by='single',
            #         # method='coverage',
            #         split_out=8
            #     )
            #     .compute()
            #     .pivot_table(
            #         values='geometry',
            #         index='xtile ytile'.split(),
            #         columns='feature',
            #         aggfunc='first',
            #     )
            #     .pipe(self.__class__)
            # )
            import shapely

            result = (
                pd.concat(frames, copy=False)
                .pipe(gpd.GeoDataFrame)
                .explode()
                .groupby(cols, sort=False)
                .geometry.agg(shapely.MultiPolygon)
                .to_frame('geometry')
                .pivot_table(
                    values='geometry',
                    index='xtile ytile'.split(),
                    columns='feature',
                    aggfunc='first'
                )
                .pipe(self.__class__)
            )

        for col in result.columns:
            result[col].set_crs(
                4326,
                inplace=True,
                allow_override=True
            )

        instance.__dict__[self.__name__] = result

    else:
        result = instance.__dict__[self.__name__]

    result.vectiles = instance
    return result


class Polygons(
    GeoDataFrameFixed
):
    vectiles: VecTiles = None
    locals().update(
        __get__=__get__,
    )

    sidewalk: gpd.GeoSeries
    road: gpd.GeoSeries
    crosswalk: gpd.GeoSeries

    # @property
    # def sidewalk(self) -> gpd.GeoSeries:
    #     re
    #
    # @property
    # def road(self) -> Self:
    #     return self[['road']]
    #
    # @property
    # def crosswalk(self) -> Self:
    #     return self[['crosswalk']]



    def explore(
            self,
            *args,
            tiles: str = 'cartodbdark_matter',
            m: folium.Map | None = None,
            dash: str = '5, 20',
            simplify=None,
            **kwargs,
    ) -> folium.Map:
        import folium
        feature2color = cfg.label2color

        loc = self.dtypes == 'geometry'
        for col in self.columns[loc]:
            color = feature2color.get(col)
            m = explore(
                self,
                geometry=col,
                *args,
                color=color,
                name=col,
                tiles=tiles,
                simplify=simplify,
                m=m,
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m
