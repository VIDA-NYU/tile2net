from __future__ import annotations

import os
from typing import *

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely import *

if False:
    from .raster import Raster
    from .grid import Tile


def deg2num_vectorized(lat_deg, lon_deg, zoom):
    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n).astype(int)
    ytile = ((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n).astype(int)
    return xtile, ytile


def num2deg_vectorized(xtile, ytile, zoom):
    print('⚠️AI GENERATED🤖')
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = np.degrees(lat_rad)
    return lat_deg, lon_deg


class Frame(
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
            raster: Raster
    ) -> Self:
        tiles: Union[
            Iterable[Tile],
            np.ndarray
        ] = raster.tiles.ravel()
        tw = np.fromiter((
            tile.xtile
            for tile in tiles
        ), int, len(tiles))
        tn = np.fromiter((
            tile.ytile
            for tile in tiles
        ), int, len(tiles))
        r = np.fromiter((
            tile.position[0]
            for tile in tiles
        ), int, len(tiles))
        c = np.fromiter((
            tile.position[1]
            for tile in tiles
        ), int, len(tiles))
        idd = np.fromiter((
            tile.idd
            for tile in tiles
        ), int, len(tiles))
        size = np.fromiter((
            tile.size
            for tile in tiles
        ), int, len(tiles))
        te = tw + raster.tile_step
        ts = tn + raster.tile_step
        zoom = raster.zoom
        gn, gw = num2deg_vectorized(tw, tn, zoom)
        gs, ge = num2deg_vectorized(te, ts, zoom)
        trans = (
            pyproj.Transformer
            .from_crs(4326, 3857, always_xy=True)
            .transform
        )
        pw, ps = trans(gw, gs)
        pe, pn = trans(ge, gn)
        geometry = shapely.box(pw, ps, pe, pn)
        data = dict(
            tw=tw, tn=tn, te=te, ts=ts,
            r=r, c=c, idd=idd,
            gn=gn, gw=gw, gs=gs, ge=ge,
            pn=pn, pw=pw, ps=ps, pe=pe,
            size=size,
        )
        result = Frame(data, geometry=geometry, crs=3857)
        return result

    def idd2outfile(
            self,
            outdir: str,
            ext='png'
    ) -> pd.Series:
        """Based on an outdir, maps each IDD to an output file."""
        outdir = str(outdir)
        sep = os.sep
        data = [
            f'{outdir}{sep}{c}_{r}_{idd}.{ext}'
            for idd, r, c in self[['idd', 'r', 'c']].itertuples(index=False)
        ]
        result = pd.Series(data, self.index, 'category')
        return result
