from __future__ import annotations
from ..cfg import cfg

import pandas as pd

from ..explore import explore

from pathlib import Path
from typing import *

from tile2net.raster.tile_utils.topology import *
from .center import Center
from .union import Union
from .features import Features
from ..fixed import GeoDataFrameFixed
from ...raster.geocode import cached

if False:
    import folium


class PedNet(
    GeoDataFrameFixed,
):
    __name__ = 'pednet'

    @property
    def feature(self) -> pd.Index:
        return self.index.get_level_values('feature')

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
    def from_polygons(
            cls,
            gdf: gpd.GeoDataFrame,
            *,
            distance: float = 5.,
            crs: int = 3857,
    ) -> Self:
        result = (
            gdf
            .to_crs(crs)
            .pipe(cls)
        )

        if distance:
            # difference(above), meaning cannot cross features above;
            # sidewalk < road < crosswalk;
            # sidewalk cannot cross road, but crosswalk can
            loc = result.feature.isin(cfg.polygon.borders)
            border = result.loc[loc]
            # todo: drop any buffer that intersects with a different feature
            #   we want roads and sidewalks to have a touching edge
            buffer = (
                result
                .loc[~loc]
                .buffer(distance=distance)
            )
            union = buffer.union_all()
            erode = union.buffer(-distance)
            intersection: gpd.GeoSeries = (
                buffer
                .intersection(erode)
                # can't cross features above
                # .difference(result.features.above, align=True)
                # unpack any resulting multipolygons from the border splitting BUE results
                .explode()
            )
            # tod: can't cross any already existing features
            # necessary to drop any islands that created from crossing the border
            features = result.features.loc[intersection.index]
            loc = intersection.intersects(features, align=False)
            intersection = intersection.loc[loc]
            concat = intersection.geometry, border.geometry
            geometry: gpd.GeoSeries = pd.concat(concat)
            result = cls(geometry=geometry)

        return result

    @classmethod
    def from_parquet(
            cls,
            path: Union[str, Path],
            distance: float = 5.,
            crs=3857,
    ) -> Self:
        """
        Load a PedNet from a parquet file.
        """
        result = (
            gpd.read_parquet(path)
            .pipe(
                cls.from_polygons,
                crs=crs,
                distance=distance,
            )
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

    def visualize(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
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
