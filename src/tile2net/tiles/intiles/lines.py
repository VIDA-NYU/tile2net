from __future__ import annotations
import shapely
from ..explore import explore
from tile2net.tiles.cfg.logger import logger
from ..cfg import cfg

from typing import *

import pandas as pd
from geopandas import GeoSeries
from shapely import *

from ..fixed import GeoDataFrameFixed

if False:
    from .intiles import InTiles
    import folium

"""
How to reconnect lines?

decompose
find all points on borders
generate pairs based on nearest neighbors
compute centroid of pairs
snap to centroids
regenerate line geometries
"""


def __get__(
        self: Lines,
        instance: InTiles,
        owner: type[InTiles]
) -> Lines:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        borders = instance.vectiles.geometry

        geometry = self.intiles.vectiles.lines
        # geom_cols = [c for c in gdf.columns if gdf[c].dtype.name == 'geometry']
        # geocols = [
        #     c for c in geometry.columns
        #     if geometry[c].dtype.name == 'geometry'
        # ]
        #
        loc = geometry.dtypes

        #
        # coords = shapely.get_coordinates(geometry)
        # repeat = shapely.get_num_points(geometry)
        # index = geometry.index.repeat(repeat)
        # result = pd.DataFrame(coords, index, 'x y'.split())


    result.intiles = instance
    return result

    return result


class Lines(
    GeoDataFrameFixed
):

    __name__ = 'lines'
    intiles: InTiles
    locals().update(
        __get__=__get__,
    )

    def explore(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            road_color: str = 'green',
            crosswalk_color: str = 'blue',
            sidewalk_color: str = 'red',
            simplify: float = None,
            dash='5, 20',
            attr: str = None,
            **kwargs,
    ) -> folium.Map:
        import folium
        _ = self.road.polygons, self.road.lines

        # m = explore(
        #     self,
        #     geometry='road.polygons',
        #     *args,
        #     color=road_color,
        #     name=f'road.polygons',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     style_kwds=dict(
        #         fill=True,
        #         fillColor=road_color,
        #         fillOpacity=0.05,
        #         weight=0,  # no stroke
        #     ),
        #     highlight=False,
        #     **kwargs,
        # )
        # m = explore(
        #     self,
        #     geometry='sidewalk.polygons',
        #     *args,
        #     color=sidewalk_color,
        #     name=f'sidewalk.polygons',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     style_kwds=dict(
        #         fill=True,
        #         fillColor=sidewalk_color,
        #         fillOpacity=0.05,
        #         weight=0,  # no stroke
        #     ),
        #     highlight=False,
        #     **kwargs,
        # )
        # m = explore(
        #     self,
        #     geometry='crosswalk.polygons',
        #     *args,
        #     color=crosswalk_color,
        #     name=f'crosswalk.polygons',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     style_kwds=dict(
        #         fill=True,
        #         fillColor=crosswalk_color,
        #         fillOpacity=0.05,
        #         weight=0,  # no stroke
        #     ),
        #     highlight=False,
        #     **kwargs,
        # )

        # m = explore(
        #     self,
        #     geometry='road.lines',
        #     *args,
        #     color=road_color,
        #     name=f'road.lines',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     **kwargs,
        # )
        # m = explore(
        #     self,
        #     geometry='sidewalk.lines',
        #     *args,
        #     color=sidewalk_color,
        #     name=f'sidewalk.lines',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     **kwargs,
        # )

        m = explore(
            self.geometry,
            *args,
            color='grey',
            name=f'tiles',
            tiles=tiles,
            simplify=simplify,
            style_kwds=dict(
                fill=False,
                dashArray=dash,
            ),
            m=m,
            **kwargs,
        )
        m = explore(
            # self,
            # geometry='crosswalk.lines',
            self.crosswalk.lines.explode().rename('geometry'),
            *args,
            color=crosswalk_color,
            name=f'crosswalk.lines',
            tiles=tiles,
            simplify=simplify,
            m=m,
            **kwargs,
        )

        m = explore(
            # self.geometry.explode(),
            self.sidewalk.lines.explode().rename('geometry'),
            *args,
            color=sidewalk_color,
            name=f'sidewalk.lines',
            tiles=tiles,
            simplify=simplify,
            m=m,
            **kwargs,
        )

        folium.LayerControl().add_to(m)
        # self.crosswalk.lines
        # shapely.get_num_geometries(self)
        return m
