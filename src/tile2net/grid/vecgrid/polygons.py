from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
)
from typing import *

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
    from ..ingrid import InGrid
    import folium


class Polygons(
    FrameWrapper
):
    """
    Polygons for each vec-tile, feature, and region.

    Handles lazy-loading of concatenated polygon geometries from vecgrid tiles:
        >>> Polygons._get

    See usage:
        >>> VecGrid.polygons
    """
    vecgrid: VecGrid = None

    def _get(
            self: Polygons,
            instance: VecGrid,
            owner: type[VecGrid]
    ) -> Polygons:
        """
        Lazy-load factory method for accessing polygons for each vec-tile, feature, and region.

        Automatically reads and dissolves polygon geometries from all vecgrid
        parquet files if not already cached. Groups by feature and tile,
        then pivots into a grid-aligned dataframe.

        Returns:
            Polygons instance with MultiPolygon geometries per tile and feature

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.polygons
            Polygons:
            feature                                              crosswalk
            xtile ytile
            9915  12120  MULTIPOLYGON (((-7910483.6 5213928.8, -7910483...
            feature                                                   road
            xtile ytile
            9915  12120  MULTIPOLYGON (((-7910857.1 5214839.8, -7910844...
            feature                                               sidewalk
            xtile ytile
            9915  12120  MULTIPOLYGON (((-7910400.7 5214764, -7910400.4...
        """
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

            msg = (
                f'Dissolving {instance.__name__}.{self.__name__} '
                f'by feature and tile'
            )
            geometry = frames[0].feature.iat[0]  # any feature

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
                    .set_geometry(geometry)
                    .pipe(self.from_frame, wrapper=self)
                )

            crs = frames[0].crs
            for col in result.columns:
                result[col].set_crs(
                    crs,
                    inplace=True,
                    allow_override=True
                )

            instance.__dict__[self.__name__] = result

        else:
            result = instance.__dict__[self.__name__]

        result.vecgrid = instance
        return result

    locals().update(
        __get__=_get,
    )

    @property
    def sidewalk(self) -> Self:
        """Subset of polygons where the feature is sidewalk."""
        loc = self.frame.feature == 'sidewalk'
        result = self.loc[loc]
        return result

    @property
    def road(self) -> Self:
        """Subset of polygons where the feature is road."""
        loc = self.frame.feature == 'road'
        result = self.loc[loc]
        return result

    @property
    def crosswalk(self) -> Self:
        """Subset of polygons where the feature is crosswalk."""
        loc = self.frame.feature == 'crosswalk'
        result = self.loc[loc]
        return result

    def explore(
            self,
            *args,
            grid: str = 'cartodbdark_matter',
            m: folium.Map | None = None,
            dash: str = '5, 20',
            simplify=None,
            **kwargs,
    ) -> folium.Map:
        """Explore polygons by feature and vec-tile using folium."""
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
