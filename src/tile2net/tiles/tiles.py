from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from pathlib import Path
from typing import *

from . import util
from .explore import explore
from .fixed import GeoDataFrameFixed
from .stitch import Stitch

if False:
    import folium


class Tiles(
    GeoDataFrameFixed,
):
    @Stitch
    def stitch(self):
        # This code block is just semantic sugar and does not run.
        # Take a look at the following methods which do run:

        # stitch to a target resolution e.g. 2048 ptxels
        self.stitch.to_resolution(...)
        # stitch to a cluster size e.g. 16 tiles
        self.stitch.to_cluster(...)
        # stitch to an XYZ scale e.g. 17
        self.stitch.to_scale(...)

    @classmethod
    def from_bounds(
            cls,
            latlon: Union[
                str,
                list[float],
            ],
            zoom: int = 19,
    ) -> Self:
        """
        Create some base Tiles from bounds in lat, lon format and a
        slippy tile zoom level.
        """
        if isinstance(latlon, str):
            if ', ' in latlon:
                split = ', '
            elif ',' in latlon:
                split = ','
            elif ' ' in latlon:
                split = ' '
            else:
                raise ValueError(
                    "latlon must be a string with coordinates "
                    "separated by either ', ', ',', or ' '."
                )
            latlon = [
                float(x)
                for x in latlon.split(split)
            ]
        gn, gw, gs, ge = latlon
        gn, gs = min(gn, gs), max(gn, gs)
        gw, ge = min(gw, ge), max(gw, ge)
        tw, tn = util.deg2num(gw, gn, zoom=zoom)
        te, ts = util.deg2num(ge, gs, zoom=zoom)
        te, tw = max(te, tw), min(te, tw)
        ts, tn = max(ts, tn), min(ts, tn)
        te += 1
        ts += 1
        tx = np.arange(tw, te)
        ty = np.arange(tn, ts)
        index = pd.MultiIndex.from_product([tx, ty])
        tx = index.get_level_values(0)
        ty = index.get_level_values(1)
        result = cls.from_integers(tx, ty, zoom=zoom)
        return result

    @classmethod
    def from_integers(
            cls,
            tx,
            ty,
            zoom: int = 19,
    ) -> Self:
        """
        Construct Tiles from integer tile numbers.
        This allows for a non-rectangular grid of tiles.
        """
        # if not isinstance(tn, np.nd)
        # if isinstance(ty, (pd.Series, pd.Index)):
        #     tn = ty.values
        #     # tn = ty.values.astype('uint32')
        # if isinstance(tn, np.ndarray):
        #     tn = tn.astype('uint32')
        # else:
        #     raise TypeError
        if isinstance(ty, (pd.Series, pd.Index)):
            tn = ty.values
        elif isinstance(ty, np.ndarray):
            tn = ty
        else:
            raise TypeError('ty must be a Series, Index, or ndarray')
        tn = tn.astype('uint32')

        if isinstance(tx, (pd.Series, pd.Index)):
            tw = tx.values
        elif isinstance(tx, np.ndarray):
            tw = tx
        else:
            raise TypeError('tx must be a Series, Index, or ndarray')
        tw = tw.astype('uint32')

        te = tw + 1
        ts = tn + 1
        names = 'tx ty'.split()
        index = pd.MultiIndex.from_arrays([tx, ty], names=names)
        # gn, gw = util.num2deg(tn, tw, zoom=zoom)
        # gs, ge = util.num2deg(ts, te, zoom=zoom)
        gw, gn = util.num2deg(tw, tn, zoom=zoom)
        ge, gs = util.num2deg(te, ts, zoom=zoom)
        geometry = shapely.box(gw, gs, ge, gn, )
        data = dict(
            gw=gw,
            gs=gs,
            ge=ge,
            gn=gn,
        )
        result = cls(
            data=data,
            geometry=geometry,
            index=index,
            crs=4326,
        )
        result.attrs['zoom'] = zoom
        return result

    def with_indir(
            self,
            indir: Union[str, Path],
    ) -> Self:
        """
        Assign an input directory to the tiles,
        """
        # todo: determine the resolution by reading an input file

    def with_outdir(
            self,
            outdir: Union[str, Path],
    ) -> Self:
        """
        Assign an output directory to the tiles.
        The tiles are saved to the output directory.
        """
        self.outdir = outdir

    def with_source(
            self,
            source,
            indir: Union[str, Path] = None,
    ) -> Self:
        """
        Assign a source to the tiles. The tiles are downloaded from
        the source and saved to an input directory.
        """

    @property
    def tx(self):
        """Tile integer X"""
        return self.index.get_level_values('tx')

    @property
    def ty(self):
        """Tile integer Y"""
        return self.index.get_level_values('ty')

    @property
    def zoom(self) -> int:
        """Tile zoom level"""
        try:
            return self.attrs['zoom']
        except KeyError:
            raise NotImplementedError('write better warning')

    @zoom.setter
    def zoom(self, value: int):
        if not isinstance(value, int):
            raise TypeError('Zoom must be an integer')
        self.attrs['zoom'] = value

    @property
    def r(self):
        ...

    @property
    def c(self):
        ...

    @property
    def resolution(self) -> int:
        """Tile resolution; inferred from input files"""
        try:
            return self.attrs['resolution']
        except KeyError as e:
            msg = (
                f'Resolution has not yet been set. To set the resolution, '
                f'you must call `tiles.with_indir()` or '
                f'`tiles.with_source()` to set the files from which the '
                f' resolution can be determined.'
            )
            raise KeyError(msg) from e

    @resolution.setter
    def resolution(self, value: int):
        self.attrs['resolution'] = value

    @property
    def tscale(self) -> int:
        """
        Tile scale; the XYZ scale of the tiles.
        Higher value means smaller area.
        """
        return self.attrs.setdefault('tscale', self.zoom)

    @property
    def file(self) -> pd.Series:
        ...

    @property
    def outdir(self):
        try:
            return self.attrs['outdir']
        except KeyError as e:
            msg = (
                f'Tiles.outdir has not been set. '
                f'You must call `Tiles.with_outdir()` to set the output '
                f'directory for the tiles.'
            )
            raise ValueError(msg) from e

    @property
    def indir(self):
        try:
            return self.attrs['indir']
        except KeyError as e:
            msg = (
                f'Tiles.indir has not been set. '
                f'You must call `Tiles.with_indir()` to set the input '
                f'directory for the tiles.'
            )
            raise ValueError(msg) from e

    @property
    def stitched(self):
        if 'stitched' not in self.attrs:
            msg = (
                f'Tiles must be stitched using `Tiles.stitch` for '
                f'example `Tiles.stitch.to_resolution(2048)` or '
                f'`Tiles.stitch.to_cluster(16)`'
            )
            raise ValueError(msg)
        result = self.attrs['stitched']
        return result

    @stitched.setter
    def stitched(self, value: Tiles):
        if not isinstance(value, Tiles):
            msg = """Tiles.stitched must be a Tiles object"""
            raise TypeError(msg)
        self.attrs['stitched'] = value


    def explore(
            self,
            *args,
            loc=None,
            tile='grey',
            subset='yellow',
            tiles: str = 'cartodbdark_matter',
            m=None,
            **kwargs
    ) -> folium.map:
        import folium
        if loc is None:
            m = explore(
                self,
                color=tile,
                name='lines',
                *args,
                **kwargs,
                tiles=tiles,
                m=m,
                style_kwds=dict(
                    # fill=False,
                    fillOpacity=0,
                )
            )
        else:
            tiles = self.loc[loc]
            loc = ~self.index.isin(tiles.index)
            m = explore(
                self.loc[loc],
                color=tile,
                name='tiles',
                *args,
                **kwargs,
                tiles=tiles,
                m=m,
                style_kwds=dict(
                    fill=False,
                    fillOpacity=0,
                )
            )
            m = explore(
                tiles,
                color=subset,
                name='subset',
                *args,
                **kwargs,
                tiles=tiles,
                m=m,
                style_kwds=dict(
                    fill=False,
                    fillOpacity=0,
                )
            )

        folium.LayerControl().add_to(m)
        return m

if __name__ == '__main__':
    """
    .stitch.to_cluster
    .stitch.to_resolution
    .stitch()
    
    tiles.with_indir().with_outdir
    
    tiles = 
    ( Tiles
        .from_bounds(...)
        .with_indir(...)
        .with_outdir(...)
        .with_stitch.to_resolution(...)
    )
    
    """
