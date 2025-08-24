from __future__ import annotations

from functools import *
from typing import Self

import geopandas as gpd
import numpy as np
import shapely.wkt
from centerline.geometry import Centerline
from tqdm import tqdm

from tile2net.grid.cfg.logger import logger
from . import mintrees, stubs
from .standalone import Lines
from ..benchmark import benchmark
from ..cfg import cfg
from ..frame.framewrapper import FrameWrapper

_ = mintrees, stubs
from ..explore import explore

if False:
    from .pednet import PedNet
    import folium




class Center(
    FrameWrapper
):

    def _get(
            self,
            instance: PedNet,
            owner: type[PedNet]
    ) -> Center:
        self: Self = FrameWrapper._get(self, instance, owner)
        cache = instance.__dict__
        key = self.__name__
        if instance is None:
            return self
        if key in cache:
            result = cache[key]
            if result.instance is not instance:
                raise NotImplementedError

        else:
            union = instance.union
            geometry = union.geometry

            msg = 'Computing centerlines'
            with benchmark(msg):
                centers = []
                it = tqdm(
                    enumerate(geometry),
                    total=len(geometry),
                    desc='centerline.geometry.Centerline()',
                )
                for i, poly in it:
                    try:
                        item = Centerline(poly).geometry
                        centers.append(item)
                    except Exception as e:
                        err = f'Centerline computation failed for index {i}:\n\t{e}'
                        logger.error(err)

            multilines = np.asarray(centers, dtype=object)
            repeat = shapely.get_num_geometries(multilines)
            lines = shapely.get_parts(multilines)
            iloc = np.arange(len(repeat)).repeat(repeat)

            msg = f'Resolving interstitial centerlines'
            with benchmark(msg):
                result = (
                    union
                    .iloc[iloc]
                    .frame
                    .set_geometry(lines)
                    .pipe(Lines.from_center)
                    .drop2nodes()
                    .frame
                    .pipe(self.from_frame, wrapper=self)
                )

            msg = f'Simplifying centerlines with tolerance 0.01'
            with benchmark(msg):
                lines = shapely.simplify(result.geometry, .01)
            # result = result.set_geometry(lines)
            # result.index.name = 'icent'

            result: Self= (
                result.frame
                .set_geometry(lines)
                .pipe(self.from_frame, wrapper=self)
            )
            result.frame.index.name = 'icent'

            cache[key] = result

        result.instance = instance
        return result

    locals().update(
        __get__=_get,
    )

    instance: PedNet = None
    __name__ = 'center'

    @cached_property
    def clipped(self) -> Self:
        msg = 'Clipping centerlines to the features'

        features = self.instance.features
        loc = ~features.feature.isin(cfg.polygon.borders)
        mutex = features.mutex.loc[loc]

        with benchmark(msg):
            # self.lines.frame
            geometry = (
                mutex
                .intersection(self.pruned.frame.union_all())
                .explode()
            )
            result = (
                gpd.GeoDataFrame(geometry=geometry, index=geometry.index)
                .pipe(self.from_frame, wrapper=self)
            )
            # result = Center(geometry=geometry, index=geometry.index)

        result.instance = self.instance
        return result

    @cached_property
    def lines(self) -> Lines:
        center = self
        lines = Lines.from_center(center.frame)
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
    def pruned(
            self,
    ) -> Self:

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

                center: Center = (
                    lines
                    .loc[loc]
                    .drop2nodes()
                    .frame
                    .pipe(Center.from_frame)
                )
                lines = center.lines
                center.instance = self.instance
                lines.pednet = self.instance
            logger.debug(f'Pruning centerlines completed after {i} iterations')

        return center

    def explore(
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
            it = features.frame.groupby(
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

        it = features.frame.groupby(level='feature', observed=False)

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
