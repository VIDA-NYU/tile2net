from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

if False:
    import folium

def explore(
        self,
        name=None,
        geometry=None,
        *args,
        **kwargs
) -> folium.Map:
    """Convenience wrapper to GeoDataFrame.explore"""
    import folium
    self = self.copy()
    _ = self.geometry
    if isinstance(self, pd.Series):
        # noinspection PyTypeChecker
        self = (
            self
            .reset_index()
            .rename(columns={'geometry': 'geometry'})
        )
        geometry = 'geometry'
    kwargs['tiles'] = kwargs.setdefault('tiles', 'cartodbdark_matter')
    style_kwargs = kwargs.setdefault('style_kwds', {})
    style_kwargs.setdefault('weight', 5)
    style_kwargs.setdefault('radius', 5)

    if (
            not geometry
            and isinstance(self, GeoDataFrame)
    ):
        geometry = self._geometry_column_name
    if name is None:
        name = geometry

    if not isinstance(self, gpd.GeoSeries):
        _ = self[geometry]

        loc = self.dtypes != 'geometry'
        loc &= self.dtypes != 'object'
        loc |= self.columns == geometry
        # todo: better way other than checking first row?
        try:
            is_string = np.fromiter((
                isinstance(x, str)
                for x in self.iloc[0]
            ), dtype=bool, count=len(self.columns))
            loc |= is_string
        except IndexError:
            ...

        columns = self.columns[loc]
        loc = self[geometry].notna().values
        max_zoom = kwargs.setdefault('max_zoom', 28)
        zoom_start = kwargs.setdefault('zoom_start', 14)
        if not loc.any():
            result = kwargs.get('m')
            if result is None:
                result = folium.Map(name=name, max_zoom=max_zoom, zoom_start=zoom_start)
            return result
        self = (
            self.loc[loc, columns]
            .reset_index()  # reset index so it shows in folium
            .set_geometry(geometry)
            .pipe(GeoDataFrame)
        )
    else:
        if not self.name:
            self.name = name
        self = (
            self
            .reset_index()
        )

    if 'MultiPoint' in self.geom_type.unique():
        # LayerControl doesn't work with MultiPoints
        self = self.explode(column=geometry, ignore_index=True)

    m = GeoDataFrame.explore(
        self,
        *args,
        **kwargs,
        name=name,
    )
    return m

