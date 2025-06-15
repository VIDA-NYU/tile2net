from __future__ import annotations
from typing import Self

from functools import *

import shapely.wkt
from centerline.geometry import Centerline
from tqdm import tqdm

from tile2net.logger import logger
from .standalone import  Lines
from . import mintrees, stubs
_ = mintrees, stubs
from tile2net.raster.tile_utils.geodata_utils import set_gdf_crs
from tile2net.raster.tile_utils.topology import *
from ..explore import explore
from ..fixed import GeoDataFrameFixed

if False:
    from .pednet import PedNet
    import folium


def __get__(
        self,
        instance: PedNet,
        owner: type[PedNet]
) -> Center:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        union = instance.union
        geometry = union.geometry

        warn = (
            f'High variance in polygon areas may cause progress-rate '
            f'fluctuations for centerline computation. '
        )
        logger.debug(warn)

        centers = []
        for i, poly in tqdm(
                enumerate(geometry),
                total=len(geometry),
                desc='Centerlines',
                leave=False
        ):
            try:
                centers.append(Centerline(poly).geometry)
            except Exception as e:
                err = f'Centerline computation failed for index {i}: {e}'
                tqdm.write(err)

        multilines = np.asarray(centers, dtype=object)
        repeat = shapely.get_num_geometries(multilines)
        lines = shapely.get_parts(multilines)
        iloc = np.arange(len(repeat)).repeat(repeat)

        result = (
            union
            .iloc[iloc]
            .set_geometry(lines)
            .pipe(Lines.from_frame)
            .drop2nodes
            .pipe(Center)
        )

        lines = shapely.simplify(result.geometry, .01)
        result = result.set_geometry(lines)
        result.index.name = 'icent'
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
    def cleaned(self) -> gpd.GeoDataFrame:
        swntw = self
        geometry = swntw.union_all()
        sw_modif_uni = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(geometry, crs=swntw.crs),
            index=swntw.index,
        )
        sw_modif_uni_met = set_gdf_crs(sw_modif_uni, 3857)
        sw_uni_lines = sw_modif_uni_met.explode()
        sw_uni_lines.reset_index(drop=True, inplace=True)
        sw_uni_lines.geometry = sw_uni_lines.simplify(2)
        sw_uni_lines.dropna(inplace=True)
        sw_uni_line2 = sw_uni_lines.copy()

        try:
            sw_cl1 = clean_deadend_dangles(sw_uni_line2)
            sw_extended = extend_lines(sw_cl1, 10, extension=0)

            sw_cleaned = remove_false_nodes(sw_extended)
            sw_cleaned.reset_index(drop=True, inplace=True)
            sw_cleaned.geometry = sw_cleaned.geometry.set_crs(3857)
        except:
            result = sw_uni_lines
        else:
            result = sw_cleaned
        return result

    @cached_property
    def clipped(self) -> gpd.GeoDataFrame:
        msg = 'Clipping centerlines to the features'
        logger.debug(msg)
        lines = self
        features = self.instance.features
        geometry = features.mutex
        predicate = "intersects"
        ifeat, iline = lines.sindex.query(geometry, predicate)
        features = features.iloc[ifeat]
        lines = lines.iloc[iline]
        geometry = (
            lines
            .intersection(features.mutex, align=False)
            .reset_index(drop=True)
            .geometry
        )
        result = Center({
            'feature': features.feature.values,
            'geometry': geometry,
        })
        result.index.name = 'iclip'
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
    def result(self) -> Self:
        """
        Create network from the full polygon dataset
        """
        lines = self.lines
        center = self
        while True:
            loc = ~lines.iline.isin(lines.stubs.iline)
            loc |= lines.iline.isin(lines.mintrees.iline)
            if np.all(loc):
                break
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
        it = features.groupby(level='feature', observed=False)

        lines = self
        if attr:
            lines = getattr(self, attr)
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
