from __future__ import annotations

from pathlib import Path
from typing import *

from tile2net.tiles.cfg.logger import logger
from tile2net.raster.tile_utils.topology import *
from .center import Center
from .features import Features
from .union import Union
from ..benchmark import benchmark
from ..cfg import cfg
from ..explore import explore
from ..fixed import GeoDataFrameFixed

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
            *,
            distance: float = .5,
            crs: int = 3857,
            save_original: bool = False,
    ) -> Self:
        logger.debug(f"Creating {cls.__name__} from {len(gdf)} polygon(s) at CRS {crs}")
        result = (
            gdf
            .to_crs(crs)
            .pipe(cls)
        )
        original = result


        if distance:

            loc = result.feature.isin(cfg.polygon.borders)
            border = result.loc[loc]
            ped = result.loc[~loc]

            msg = f"Dissolving and buffering {len(ped)} polygon(s) by {distance}"
            with benchmark(msg):
                buffer = (
                    ped
                    .dissolve(level='feature')
                    .buffer(distance, cap_style='flat')
                )

            n = len(ped) + len(border)
            msg = f'Computing the union of {n} geometries'
            with benchmark(msg):
                union = (
                    pd.concat([buffer.geometry, border.geometry], axis=0)
                    .pipe(gpd.GeoSeries)
                    .union_all()
                )

            n = shapely.get_num_geometries(union)
            msg = f'Eroding the union of {n} geometries by {distance}'
            with benchmark(msg):
                erode = union.buffer(-distance, cap_style='flat')

            n = shapely.get_num_geometries(buffer).sum()
            m = shapely.get_num_geometries(erode)
            msg = (
                f'Intersecting {n} buffered polygon(s) with {m} unioned '
                f'and eroded polygon(s), preventing crossing features, '
                f'and unpacking multipolygons'
            )
            with benchmark(msg):
                intersection: gpd.GeoDataFrame = (
                    buffer
                    # Intersecting buffered polygons with eroded union
                    .intersection(erode)
                    # Prevent crossing features
                    .difference(result.features.other, align=True)
                    # Unpack multipolygons
                    .explode()
                )

            # drop any islands that created from crossing the border
            msg = f'Dropping buffered geometries which crossed the border'
            with benchmark(msg):
                iped, iint = (
                    intersection.sindex
                    .query(ped.geometry, predicate='intersects')
                )
                loc = intersection.index[iint] == ped.index[iped]
                iloc = np.unique(iint[loc])
                intersection = intersection.iloc[iloc]

            geometry = (
                pd.concat([intersection.geometry, border.geometry])
                .pipe(gpd.GeoSeries)
            )
            result = cls(geometry=geometry)

        if save_original:
            msg = f'Dissolving the original polygons by feature'
            logger.debug(msg)
            result.features.original = original.dissolve(level='feature').geometry

        return result

    @classmethod
    def from_parquet(
            cls,
            path: Union[str, Path],
            distance: float = .5,
            crs=3857,
            save_original: bool = False,
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
                save_original=save_original,
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
