from __future__ import annotations
from pathlib import Path

from functools import *
from typing import Self

import shapely.wkt
from centerline.geometry import Centerline
from tqdm import tqdm

from tile2net.logger import logger
from . import mintrees, stubs
from .standalone import Lines
from ..benchmark import benchmark

_ = mintrees, stubs
from tile2net.raster.tile_utils.geodata_utils import set_gdf_crs
from tile2net.raster.tile_utils.topology import *
from ..explore import explore
from ..fixed import GeoDataFrameFixed

if False:
    from .pednet import PedNet
    import folium


def __get__(
        self: Center,
        instance: PedNet,
        owner: type[PedNet]
) -> Center:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        checkpoint = None
        if instance.checkpoint:
            checkpoint = instance.checkpoint / 'center.parquet'
        if checkpoint and checkpoint.exists():
                result = (
                    gpd.read_parquet(checkpoint)
                    .pipe(self.__class__)
                )
        else:
            union = instance.union
            geometry = union.geometry

            warn = (
                f'High variance in polygon areas may cause progress-rate '
                f'fluctuations for centerline computation. '
            )
            logger.debug(warn)

            msg = 'Computing centerlines'
            with benchmark(msg):
                centers = []
                it = tqdm(
                    enumerate(geometry),
                    total=len(geometry),
                    desc='Centerlines',
                    leave=False
                )
                for i, poly in it:
                    try:
                        item = Centerline(poly).geometry
                        centers.append(item)
                    except Exception as e:
                        err = f'Centerline computation failed for index {i}: {e}'
                        tqdm.write(err)

            multilines = np.asarray(centers, dtype=object)
            repeat = shapely.get_num_geometries(multilines)
            lines = shapely.get_parts(multilines)
            iloc = np.arange(len(repeat)).repeat(repeat)

            msg = f'Resolving interstitial centerlines'
            with benchmark(msg):
                result = (
                    union
                    .iloc[iloc]
                    .set_geometry(lines)
                    .pipe(Lines.from_frame)
                    .drop2nodes
                    .pipe(Center)
                )

            msg = f'Simplifying centerlines with tolerance 0.01'
            with benchmark(msg):
                lines = shapely.simplify(result.geometry, .01)
            result = result.set_geometry(lines)
            result.index.name = 'icent'

            if checkpoint:
                msg = f'Saving centerlines to {checkpoint}'
                logger.debug(msg)
                result.to_parquet(checkpoint)


        instance.__dict__[self.__name__] = result

    result.instance = instance
    return result


class Center(
    GeoDataFrameFixed,
):
    locals().update(
        __get__=__get__,
    )

    instance: PedNet = None
    __name__ = 'center'

    @cached_property
    def clipped(self) -> Self:
        lines = self.pruned
        features = self.instance.features
        msg = 'Clipping centerlines to the features'
        logger.debug(msg)
        geometry = features.mutex
        predicate = "intersects"
        ifeat, iline = lines.sindex.query(geometry, predicate)
        features = features.iloc[ifeat]
        lines = lines.iloc[iline]
        geometry = (
            lines
            .intersection(features.mutex, align=False)
            .set_axis(features.index)
        )
        result = Center(geometry=geometry)
        result.instance = self.instance

        return result

    @cached_property
    def lines(self) -> Lines:
        center = self
        lines = Lines.from_frame(center)
        lines.pednet = self.instance
        return lines


    @cached_property
    def crosswalk(self) -> gpd.GeoDataFrame:
        loc = self.clipped.feature == 'crosswalk'
        crosswalk = self.clipped.loc[loc].copy()
        return crosswalk

    @cached_property
    def sidewalk(self) -> gpd.GeoDataFrame:
        loc = self.clipped.feature == 'sidewalk'
        sidewalk = self.clipped.loc[loc].copy()
        return sidewalk

    @cached_property
    def pruned(self) -> Self:
        """
        Create network from the full polygon dataset
        """
        lines = self.lines
        center = self
        msg = f'Pruning centerlines to remove stubs and mintrees'
        # logger.debug(msg)
        with benchmark(msg):
            i = -1
            while True:
                i+= 1
                msg = f'Iteration {i} of pruning centerlines'
                logger.debug(msg)
                loc = ~lines.iline.isin(lines.stubs.iline)
                loc |= lines.iline.isin(lines.mintrees.iline)
                if np.all(loc):
                    break
                msg = f'Resolving interstitial lines'
                logger.debug(msg)
                center: Center = (
                    lines
                    .loc[loc]
                    .pipe(Lines)
                    .drop2nodes
                    .pipe(Center)
                )
                lines = center.lines
                center.instance = self.instance
                lines.pednet = self.instance
            msg = f'Pruning centerlines completed after {i} iterations'
            logger.debug(msg)
        return center

    @cached_property
    def pruned(
            self,
    ) -> Self:
        checkpoint = (
            self.instance.checkpoint / 'pruned.parquet'
            if self.instance.checkpoint
            else None
        )
        if checkpoint and checkpoint.exists():
            center = (
                gpd.read_parquet(checkpoint)
                .pipe(self.__class__)
            )
            center.instance = self.instance
            center.lines.pednet = self.instance
            return center

        lines = self.lines
        center = self
        msg = 'Pruning centerlines to remove stubs and mintrees'
        with benchmark(msg):
            i = -1
            while True:
                i += 1
                logger.debug(f'Iteration {i} of pruning centerlines')
                loc = ~lines.iline.isin(lines.stubs.iline)
                loc |= lines.iline.isin(lines.mintrees.iline)
                if np.all(loc):
                    break
                logger.debug('Resolving interstitial lines')
                center = (
                    lines
                    .loc[loc]
                    .pipe(Lines)
                    .drop2nodes
                    .pipe(Center)
                )
                lines = center.lines
                center.instance = self.instance
                lines.pednet = self.instance
            logger.debug(f'Pruning centerlines completed after {i} iterations')

        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with benchmark(f'Saving pruned centerlines to {checkpoint}'):
                center.to_parquet(checkpoint)

        return center

    def visualize(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            line_color='grey',
            node_color='red',
            simplify: float = None,
            dash='5, 20',
            attr: str = None,
            **kwargs,
    ) -> folium.Map:
        import folium
        features = self.instance.features
        feature2color = features.color.to_dict()
        _ = features.mutex

        if 'original' in features:
            it = features.groupby(
                level='feature',
                observed=False,
            )
            for feature, frame in it:
                color = feature2color[feature]
                m = explore(
                    frame,
                    geometry='original',
                    *args,
                    color=color,
                    name=f'{feature} (original)',
                    tiles=tiles,
                    simplify=simplify,
                    m=m,
                    style_kwds=dict(
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.05,
                        weight=0,  # no stroke
                    ),
                    highlight=False,
                    **kwargs,
                )


        it = features.groupby(level='feature', observed=False)

        for feature, frame in it:
            color = feature2color[feature]
            m = explore(
                frame,
                geometry='mutex',
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

        lines = self
        if attr:
            lines = getattr(lines, attr)
        lines = lines.reset_index()
        if 'feature' in lines.columns:
            it = lines.groupby('feature', observed=False)
            for feature, frame in it:
                color = feature2color[feature]
                m = explore(
                    frame,
                    *args,
                    color=color,
                    name=f'{feature} lines',
                    tiles=tiles,
                    simplify=simplify,
                    m=m,
                    **kwargs,
                )
        else:
            m = explore(
                lines,
                *args,
                color=line_color,
                name='centerlines',
                tiles=tiles,
                simplify=simplify,
                m=m,
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m
