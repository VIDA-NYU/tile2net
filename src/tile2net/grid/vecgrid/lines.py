
from __future__ import annotations

import geopandas as gpd


from concurrent.futures import (
    ThreadPoolExecutor,
)
from typing import Self

import geopandas as gpd
import pandas as pd
from geopandas.array import GeometryDtype
from shapely import MultiLineString

from ..benchmark import benchmark
from ..cfg import cfg
from ..explore import explore
from ..frame.framewrapper import FrameWrapper

if False:
    from .vecgrid import VecGrid
    import folium


class Lines(
    FrameWrapper,
):
    __name__ = 'lines'
    vecgrid: VecGrid = None

    def _get(
            self,
            instance: VecGrid,
            owner: type[VecGrid]
    ) -> Self:

        if instance is None:
            result = self

        elif self.__name__ not in instance.__dict__:

            idx_names = list(instance.index.names)

            def _read(idx_path):
                idx, path = idx_path
                gdf = (
                    gpd.read_parquet(path)
                    .reset_index(names='feature')
                )
                if not isinstance(idx, tuple):
                    idx = (idx,)
                for name, val in zip(idx_names, idx):
                    gdf[name] = val
                gdf['src'] = 'lines'
                return gdf

            tasks = [
                (i, p)
                for i, p in
                instance.file.lines.items()
            ]

            with ThreadPoolExecutor() as ex:
                frames = list(ex.map(_read, tasks))

            msg = f'Dissolving {instance.__name__}.{self.__name__} by feature and tile'
            cols = 'xtile ytile feature'.split()
            # gpd.GeoDataFrame.dropna(_, axis='geometry')
            #
            # frame = pd.concat(frames, copy=False)
            # frame.dropna(subset='geometry')
            # gpd.GeoDataFrame.dropna(self=_, subset='geometry')

            with benchmark(msg):
                result = (
                    pd.concat(frames, copy=False)
                    .pipe(gpd.GeoDataFrame)
                    .explode()
                    .loc[lambda s: ~s.geometry.is_empty]
                    .groupby(cols, sort=False, observed=True)
                    .geometry.agg(lambda s: MultiLineString(tuple(s)))
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
                result.frame[col].set_crs(
                    4326,
                    inplace=True,
                    allow_override=True
                )

        else:
            result = instance.__dict__[self.__name__]

        result.vecgrid = instance
        return result

    locals().update(
        __get__=_get,
    )

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

        # detect geometry-typed columns
        geom_cols = [
            col
            for col in self.columns
            if isinstance(self[col].dtype, GeometryDtype)
        ]

        for col in geom_cols:
            gdf = (
                self.frame[[col]]
                .copy()
                .set_geometry(col)
            )

            color = feature2color.get(col)
            m = explore(
                gdf,
                *args,
                color=color,
                geometry=col,
                name=col,
                tiles=tiles,
                simplify=simplify,
                m=m,
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m
