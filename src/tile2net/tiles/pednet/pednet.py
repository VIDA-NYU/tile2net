from __future__ import annotations
import geopandas as gpd
import dask_geopandas as dgpd
from dask import delayed
from pathlib import Path
import pandas as pd
import shapely
from typing import Self

from pathlib import Path
from typing import *

from tile2net.logger import logger
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
            checkpoint: str = None,
    ) -> Self:
        logger.debug(f"Creating {cls.__name__} from {len(gdf)} polygon(s) at CRS {crs}")
        result = (
            gdf
            .to_crs(crs)
            .pipe(cls)
        )
        original = result
        CHECKPOINT = checkpoint
        result.checkpoint = checkpoint

        if result.checkpoint:
            checkpoint = result.checkpoint / 'pednet.parquet'
            if checkpoint.exists():
                result = (
                    gpd.read_parquet(checkpoint)
                    .pipe(cls)
                )
                return result

        if distance:
            loc = result.feature.isin(cfg.polygon.borders)
            border = result.loc[loc, 'geometry']
            ped = result.loc[~loc, 'geometry']

            msg = f"Buffering {len(ped)} polygon(s) by {distance}"
            logger.debug(msg)
            buffer = (
                result
                .loc[~loc]
                .buffer(distance=distance, cap_style='flat')
            )

            n = len(buffer) + len(border)
            msg = f'Computing the union of {n} geometries'
            # logger.debug(msg)
            with benchmark(msg):
                union = (
                    pd.concat([buffer, border], axis=0)
                    .pipe(gpd.GeoSeries)
                    .union_all()
                )
            n = shapely.get_num_geometries(union)
            msg = f'Eroding the union of {n} geometries by {distance}'
            # logger.debug(msg)
            with benchmark(msg):
                erode = union.buffer(-distance, cap_style='flat')

            loc = ~buffer.index.isin(cfg.polygon.borders)
            msg = f"Intersecting {len(buffer)} buffered polygon(s) with eroded union"
            # logger.debug(msg)
            with benchmark(msg):
                intersection: gpd.GeoSeries = (
                    buffer
                    # only expand non-borders
                    .loc[loc]
                    .intersection(erode)
                    # can't cross other features
                    .difference(result.features.other, align=True)
                    # unpack any resulting multipolygons from the border splitting BUE results
                    .explode()
                )

            # drop any islands that created from crossing the border
            msg = f'Dropping buffered geometries which crossed the border'
            # logger.debug(msg)
            with benchmark(msg):
                features = result.features.loc[intersection.index]
                loc = intersection.intersects(features, align=False)
                intersection = intersection.loc[loc]

            geometry = (
                pd.concat([intersection, border])
                .pipe(gpd.GeoSeries)
            )
            result = cls(geometry=geometry)
            result.checkpoint = CHECKPOINT

        if save_original:
            msg = f'Dissolving the original polygons by feature'
            logger.debug(msg)
            result.features.original = original.dissolve(level='feature').geometry

        if result.checkpoint:
            checkpoint = result.checkpoint / 'pednet.parquet'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            msg = f'Saving {len(result)} polygons to {checkpoint}'
            with benchmark(msg):
                result.to_parquet(checkpoint, index=False)

        return result

    @classmethod
    def from_parquet(
            cls,
            path: Union[str, Path],
            distance: float = .5,
            crs=3857,
            checkpoint: str = None,
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
                checkpoint=checkpoint,
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

    @property
    def checkpoint(self) -> Path:
        return self.attrs.setdefault('checkpoint', None)

    @checkpoint.setter
    def checkpoint(self, value: str | Path | None):
        if value is None:
            self.attrs.pop('checkpoint', None)
        else:
            self.attrs['checkpoint'] = Path(value).resolve()
