from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import *

import geopandas as gpd
import pandas as pd
from geopandas.array import GeometryDtype
from shapely import MultiLineString

from ..benchmark import benchmark
from ..cfg import cfg
from ..explore import explore
from ..frame.framewrapper import FrameWrapper

if TYPE_CHECKING:
    from .vecgrid import VecGrid
    from ..grid import Grid
    import folium


class Network(
    FrameWrapper,
):
    """
    Network for each vec-tile, feature, and region.

    Handles lazy-loading of concatenated line geometries from vecgrid tiles:
        >>> Network.__get__

    See usage:
        >>> VecGrid.network
    """
    __name__ = 'network'
    vecgrid: VecGrid = None



    def __get__(
            self,
            instance: VecGrid,
            owner: type[VecGrid]
    ) -> Self:
        """
        Lazy-load factory method for accessing network for each vec-tile, feature, and region.

        Automatically reads and dissolves line geometries from all vecgrid
        parquet files if not already cached. Groups by feature and tile,
        then pivots into a grid-aligned dataframe.

        Returns:
            Network instance with MultiNetworktring geometries per tile and feature

        Example:
            >>> grid: Grid
            >>> grid.vecgrid.network
            Network:
            feature                                              crosswalk
            xtile ytile
            9915  12120  MULTILINESTRING ((-7910926 5213692.6, -7910925...
            feature                                               sidewalk
            xtile ytile
            9915  12120  MULTILINESTRING ((-7910947.3 5213616.8, -79109...
        """

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
                gdf['src'] = 'network'
                return gdf

            tasks = [
                (i, p)
                for i, p in
                instance.file.network.items()
            ]

            with ThreadPoolExecutor() as ex:
                frames = list(ex.map(_read, tasks))

            msg = f'Dissolving {instance.__name__}.{self.__name__} by feature and tile'
            cols = 'xtile ytile feature'.split()
            geometry = frames[0].feature.iat[0]  # any feature

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
                    .set_geometry(geometry)
                    .pipe(self.from_frame, wrapper=self)
                )

            crs = frames[0].crs
            for col in result.columns:
                result.frame[col].set_crs(
                    crs,
                    inplace=True,
                    allow_override=True
                )

        else:
            result = instance.__dict__[self.__name__]

        result.vecgrid = instance
        return result

    @property
    def sidewalk(self) -> Self:
        """Subset of network where the feature is sidewalk."""
        loc = self.frame.feature == 'sidewalk'
        result = self.loc[loc]
        return result

    @property
    def road(self) -> Self:
        """Subset of network where the feature is road."""
        loc = self.frame.feature == 'road'
        result = self.loc[loc]
        return result

    @property
    def crosswalk(self) -> Self:
        """Subset of network where the feature is crosswalk."""
        loc = self.frame.feature == 'crosswalk'
        result = self.loc[loc]
        return result

    def explore(
            self,
            *args,
            tiles: str = 'cartodbdark_matter',
            m: folium.Map | None = None,
            simplify=None,
            **kwargs,
    ) -> folium.Map:
        """Explore network by feature and vec-tile using folium."""
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
