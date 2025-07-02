from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from ..cfg import cfg
from ..explore import explore
from ..fixed import GeoDataFrameFixed

if False:
    from .intiles import InTiles
    import folium

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

        lines: gpd.GeoDataFrame = instance.vectiles.lines.copy()
        lines.columns = lines.columns.str.removeprefix('lines.')
        cols = lines.dtypes == 'geometry'
        result: gpd.GeoDataFrame = (
            lines
            .loc[:, cols]
            .stack(future_stack=True)
            .rename('geometry')
            .dropna()
            .explode()
            .reset_index()
            .rename(columns=dict(level_2='feature', ))
            .pipe(self.__class__)
        )
        instance.__dict__[self.__name__] = result

        COORDS = shapely.get_coordinates(result.geometry)
        repeat = shapely.get_num_points(result.geometry)
        indices = iline = result.index.repeat(repeat)

        unique, ifirst, repeat = np.unique(
            iline,
            return_counts=True,
            return_index=True
        )

        borders = instance.vectiles.exterior.union_all()
        istop = ifirst + repeat
        ilast = istop - 1

        iend = np.concatenate([ifirst, ilast])
        iline = iline[iend]
        frame = result.loc[iline].reset_index()
        coords = COORDS[iend]
        geometry = shapely.points(coords)
        borders = instance.vectiles.exterior.union_all()
        loc = shapely.intersects(geometry, borders)
        iend = iend[loc]
        coords = COORDS[iend]
        frame = frame.loc[loc]

        tree = cKDTree(coords)
        ileft = np.arange(len(coords))
        dists, iright = tree.query(coords, 2, workers=-1)
        iright = iright[:, 1]
        arange = np.arange(len(coords))
        arrays = arange, iright

        needles = pd.MultiIndex.from_arrays(arrays)
        haystack = needles.swaplevel()

        loc = frame.xtile.values[ileft] != frame.xtile.values[iright]
        loc |= frame.ytile.values[ileft] != frame.ytile.values[iright]
        loc &= needles.isin(haystack)
        ileft = ileft[loc]
        iright = iright[loc]
        left = coords[ileft]
        right = coords[iright]
        mean = (left + right) / 2
        ileft = iend[ileft]
        iright = iend[iright]
        COORDS[ileft] = mean
        COORDS[iright] = mean

        geometry = shapely.linestrings(COORDS, indices=indices)

        geometry = (
            result
            .set_geometry(geometry)
            .dissolve('feature')
            .line_merge()
            .explode()
        )
        result = self.__class__(geometry=geometry)


    result.intiles = instance
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
            tile_color='grey',
            simplify: float = None,
            dash='5, 20',
            attr: str = None,
            **kwargs,
    ) -> folium.Map:
        import folium
        feature2color = cfg.label2color
        if tile_color:
            m = explore(
                self.intiles.vectiles.boundary,
                *args,
                color='grey',
                name='tile',
                tiles=tiles,
                simplify=simplify,
                style_kwds=dict(
                    fill=False,
                    dashArray=dash,
                ),
                m=m,
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
                    name=feature,
                    tiles=tiles,
                    simplify=simplify,
                    m=m,
                    **kwargs,
                )

        folium.LayerControl().add_to(m)
        return m
