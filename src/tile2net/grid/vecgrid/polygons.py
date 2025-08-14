from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
)

import geopandas as gpd
import pandas as pd
import shapely

from ..benchmark import benchmark
from ..cfg import cfg
from ..cfg.logger import logger
from ..explore import explore
from ..frame.framewrapper import FrameWrapper

if False:
    from .vecgrid import VecGrid
    import folium


class Polygons(
    FrameWrapper
):
    vecgrid: VecGrid = None

    def _get(
            self: Polygons,
            instance: VecGrid,
            owner: type[VecGrid]
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

            msg = f'Dissolving {instance.__name__}.{self.__name__} by feature and tile'

            cols = 'xtile ytile feature'.split()
            with benchmark(msg):

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
                    .reindex(instance.index)
                    .pipe(self.from_frame, wrapper=self)
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

        result.vecgrid = instance
        return result

    sidewalk: gpd.GeoSeries
    road: gpd.GeoSeries
    crosswalk: gpd.GeoSeries

    locals().update(
        __get__=_get,
    )


    def explore(
            self,
            *args,
            grid: str = 'cartodbdark_matter',
            m: folium.Map | None = None,
            dash: str = '5, 20',
            simplify=None,
            **kwargs,
    ) -> folium.Map:
        import folium
        feature2color = cfg.label2color

        loc = self.frame.dtypes == 'geometry'
        for col in self.columns[loc]:
            color = feature2color.get(col)
            m = explore(
                self.frame,
                geometry=col,
                *args,
                color=color,
                name=col,
                grid=grid,
                simplify=simplify,
                m=m,
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m
