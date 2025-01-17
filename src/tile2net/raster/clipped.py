from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import *

import geopandas as gpd
import pandas as pd
from shapely import *

from tile2net.raster.util import read_file

if False:
    from .raster import Raster
    import folium


class Clipped(
    gpd.GeoDataFrame
):
    """Frame encapsulating the tiles of the Raster"""

    # tile numbers
    gn: pd.Series[float]
    gw: pd.Series[float]
    gs: pd.Series[float]
    ge: pd.Series[float]

    tn: pd.Series[int]
    tw: pd.Series[int]
    te: pd.Series[int]
    ts: pd.Series[int]

    pn: pd.Series[float]
    pw: pd.Series[float]
    ps: pd.Series[float]
    pe: pd.Series[float]

    r: pd.Series[int]
    c: pd.Series[int]
    idd: pd.Series[int]

    geometry: gpd.GeoSeries[Polygon]
    size: pd.Series[float]

    @classmethod
    def from_raster(
            cls,
            self: Raster,
            infiles: Union[
                # filename
                str,
                    # kwargs
                dict[str, Any],
                    # list of filenames or kwargs
                list[Union[
                    str,
                    dict[str, Any],
                ]],
            ],
            annotations: Union[
                str,
                list[str],
                Callable[[gpd.GeoDataFrame], pd.Series],
                list[Callable[[gpd.GeoDataFrame], pd.Series]],
            ],
            colors: dict[str, str],
            outdir=None,
            zorder: dict[str, int] = None,
            zorder_default=20,
    ):
        """
        infiles:
            path or paths to geodataframes of annotations
        annotations:
            str:
                label name of all geometry in the GDF
            list[str]:
                label name for all geometry in each respective GDF
            Callable[[gpd.GeoDataFrame], pd.Series]:
                function that takes GDF and returns series of labels
                e.g. lambda gdf: gdf['f_type']
            list[Callable[[gpd.GeoDataFrame], pd.Series]]:
                list of functions that take GDF and return series of labels
                for each respective GDF
                e.g. [lambda gdf: gdf['f_type'], lambda gdf: gdf['label']]
        colors:
            dictionary mapping annotations to their colors
        outdir:
            output directory for annotations; tempdir by default
        zorder:
            dictionary mapping annotations to their zorder
        zorder_default:
            default zorder for annotations
        """

        if not isinstance(infiles, list):
            infiles = [infiles]
        if not isinstance(annotations, list):
            annotations = [annotations]

        if outdir is None:
            outdir = tempfile.gettempdir()
        outdir = f'{outdir}{os.sep}annotations'
        os.makedirs(outdir, exist_ok=True)

        def submit(annotation, infile):
            if isinstance(infile, dict):
                gdf = read_file(**infile)
            else:
                gdf = read_file(infile)
            if isinstance(annotation, str):
                series = pd.Series(annotation, index=gdf.index)
            elif isinstance(annotation, Callable):
                series = annotation(gdf)
            else:
                raise ValueError('annotation must be a string or a callable')
            loc = series.notna()
            return (
                gdf.geometry
                .pipe(gpd.GeoDataFrame)
                .assign(annotation=series.values)
                .loc[loc]
            )

        threads = ThreadPoolExecutor()
        it = threads.map(submit, annotations, infiles)
        crs = self.frame.crs
        concat = [
            gdf.to_crs(crs)
            for gdf in it
        ]
        vector = (
            pd.concat(concat)
            .pipe(gpd.GeoDataFrame)
        )
        vector['color'] = (
            pd.Series(colors)
            .loc[vector['annotation']]
            .values
        )
        vector['zorder'] = (
            pd.Series(zorder)
            .reindex(vector['annotation'], fill_value=zorder_default)
            .values
        )
        predicate = 'intersects'
        iright, ileft = vector.sindex.query(
            self.frame.geometry,
            predicate=predicate,
        )
        tiles = self.frame.iloc[iright]
        idd = (
            tiles['idd']
            .astype('category')
            .values
        )
        vector = vector.iloc[ileft]
        geometry = vector.geometry.intersection(tiles.geometry, align=False)
        size = tiles['size'].values
        outfiles = (
            self.frame
            .idd2outfile(outdir)
            .loc[tiles.index]
            .values
        )
        clippeds = vector.assign(
            geometry=geometry,
            size=size,
            idd=idd,
            outfiles=outfiles,
        )
        tiles = self.frame
        outfiles = (
            self.frame
            .idd2outfile(outdir)
            .loc[tiles.index]
            .values
        )
        tiles['outfile'] = outfiles
        result = clippeds.pipe(Clipped)
        result.tiles = tiles
        return result

    # def explore(
    #         self,
    #         *args,
    #         tiles='cartodbdark_matter',
    #         **kwargs
    # ) -> folium.Map:
    #     return super().explore(*args, tiles=tiles, **kwargs)

