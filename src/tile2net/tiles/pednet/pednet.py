from __future__ import annotations

import pandas as pd

from ..explore import explore

from pathlib import Path
from typing import *

from tile2net.raster.tile_utils.topology import *
from .center import Center
from .union import Union
from .features import Features
from ..fixed import GeoDataFrameFixed

if False:
    import folium

class PedNet(
    GeoDataFrameFixed,
):
    __name__ = 'pednet'
    feature: pd.Series

    @classmethod
    def from_polygons(
            cls,
            gdf: gpd.GeoDataFrame,
            crs=3857,
    ) -> Self:
        result = (
            gdf
            .to_crs(crs)
            .pipe(cls)
        )
        return result

    @classmethod
    def from_parquet(
            cls,
            path: Union[str, Path],
            crs=3857,
    ) -> Self:
        """
        Load a PedNet from a parquet file.
        """
        result = (
            gpd.read_parquet(path)
            .pipe(cls.from_polygons, crs=crs)
        )
        return result

    @Union
    def union(self):
        # This code block is just semantic sugar and does not run.
        _ = self.union.polygons
        _ = self.union.centerlines

    @Features
    def features(self):
        # This code block is just semantic sugar and does not run.
        _ = self.features.polygons
        _ = self.features.centerlines

    @Center
    def center(self):
        # This code block is just semantic sugar and does not run.
        _ = self.center
        _ = self.center.sidewalk
        _ = self.center.crosswalk

    def explore(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            line='grey',
            node='red',
            simplify: float = None,
            dash='5, 20',
            **kwargs,
    ) -> folium.Map:
        features = self.features
        feature2color = features.color.to_dict()
        it = self.groupby('feature', observed=False)

        for feature, frame in it:
            color = feature2color[feature]
            m = explore(
                frame,
                *args,
                color=color,
                name=f'{feature} polygons',
                tiles=tiles,
                simplify=simplify,
                m=m,
                style_kwds=dict(
                    fill=False,
                    dashArray=dash,
                ),
                **kwargs,
            )

        import folium
        folium.LayerControl().add_to(m)
        return m
