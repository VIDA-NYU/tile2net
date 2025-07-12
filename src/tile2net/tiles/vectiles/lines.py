
from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.array import GeometryDtype

from ..cfg import cfg
from ..explore import explore
from ..fixed import GeoDataFrameFixed

if False:
    from .vectiles import VecTiles
    import folium


def __get__(
        self: Lines,
        instance: VecTiles,
        owner: type[VecTiles]
) -> Lines:

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

        tasks = [(i, p) for i, p in instance.file.lines.items()]

        with ThreadPoolExecutor() as ex:
            frames = list(ex.map(_read, tasks))

        table = (
            pd.concat(frames, copy=False)
            .pivot_table(
                values='geometry',
                index=idx_names,
                columns=['feature', 'src'],
                aggfunc='first'
            )
            .reindex(instance.index)
        )

        table.columns = [f'lines.{feat}' for feat, _ in table.columns]

        gdf = gpd.GeoDataFrame(table)
        target_crs = getattr(instance, 'crs', None)

        for col in gdf.columns:
            if isinstance(gdf[col].dtype, GeometryDtype):
                gdf[col].set_crs(4326, inplace=True, allow_override=True)
                if target_crs and target_crs != 4326:
                    gdf[col] = gdf[col].to_crs(target_crs)

        instance[gdf.columns] = gdf

        loc = instance.columns.str.startswith('lines.')
        loc |= instance.columns == 'feature'
        result = instance.loc[:, loc]

        result = self.__class__(result)
        regex = '^lines\\.'
        columns = result.columns.str.replace(regex, '', regex=True)
        result.columns = columns
        instance.__dict__[self.__name__] = result

    else:
        result = instance.__dict__[self.__name__]

    result.vectiles = instance
    return result

class Lines(
    GeoDataFrameFixed
):
    __name__ = 'lines'
    vectiles: VecTiles = None
    locals().update(
        __get__=__get__,
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
                self[[col]]
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
