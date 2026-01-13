from __future__ import annotations
from typing import *

import io
import os
from functools import *
from math import ceil
from pathlib import Path
from typing import overload

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from PIL import Image
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

import tile2net.grid.pednet.lines
from ..cfg import cfg
from ..cfg.logger import logger
from ..explore import explore
from ..frame.framewrapper import FrameWrapper

if False:
    from .ingrid import InGrid
    import folium


class Network(
    FrameWrapper
):
    """
    Network for each feature and region, dissolved across tiles.

    Handles lazy-loading of concatenated network from vecgrid tiles:
        >>> Network.__get__

    See usage:
        >>> InGrid.network
    """
    __name__ = 'network'


    def _get(
            self,
            instance: InGrid,
            owner: type[InGrid]
    ) -> Self:
        """
        Lazy-load factory method for accessing network for each feature, dissolved across tiles.

        Automatically concatenates and dissolves line geometries from all
        vec-tiles if not already cached. Stitches line segments at tile
        boundaries and removes 2-degree nodes. Results are saved as parquet
        for persistence across sessions.

        Returns:
            Network instance with dissolved features from all vecgrid tiles

        Example:
            >>> ingrid: InGrid
            >>> ingrid.network
            Network:
                                                        geometry
            feature
            crosswalk  LINESTRING (-7910926 5213692.6, -7910925.6 521...
            crosswalk  LINESTRING (-7910894.8 5213730.1, -7910895 521...
            sidewalk   LINESTRING (-7910885.6 5213961.8, -7910885.6 5...
        """
        self = super()._get(instance, owner)
        if instance is None:
            raise NotImplementedError

        key = self.__name__
        cache = instance.frame.__dict__

        if key in cache:
            result = cache[key]
        else:
            file = self.file
            Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
            if os.path.exists(file):
                msg = (
                    f'Loading {owner.__qualname__}.{self.__name__} '
                    f'from \n\t{file}'
                )
                logger.info(msg)
                result = (
                    gpd.read_parquet(file)
                    .pipe(self.__class__.from_frame, wrapper=self)
                )
                cache[key] = result
            else:
                network: gpd.GeoDataFrame = instance.vecgrid.network.frame.copy()
                with instance.line_benchmark:

                    network.columns = (
                        network.columns.str
                        .removeprefix('network.')
                    )
                    cols = network.dtypes == 'geometry'

                    msg = f"Stacking geometric columns into a single geometry column."
                    logger.debug(msg)
                    result: Network = (
                        network
                        .loc[:, cols]
                        .stack(future_stack=True)
                        .rename('geometry')
                        .set_crs(network.crs)
                        .dropna()
                        .explode()
                        .reset_index()
                        .rename(columns=dict(level_2='feature', ))
                        .pipe(self.__class__.from_frame, wrapper=self)
                    )

                    cache[key] = result

                    msg = "Finding coordinates that intersect tile boundaries"
                    debug = logger.debug(msg)
                    COORDS = shapely.get_coordinates(result.geometry)
                    repeat = shapely.get_num_points(result.geometry)
                    indices = iline = result.index.repeat(repeat)

                    unique, ifirst, repeat = np.unique(
                        iline,
                        return_counts=True,
                        return_index=True
                    )

                    istop = ifirst + repeat
                    ilast = istop - 1

                    iend = np.concatenate([ifirst, ilast])
                    iline = iline[iend]
                    frame = result.loc[iline].frame.reset_index()
                    coords = COORDS[iend]
                    geometry = shapely.points(coords)
                    borders = instance.vecgrid.geometry.exterior.union_all()
                    loc = shapely.intersects(geometry, borders)
                    iend = iend[loc]
                    coords = COORDS[iend]
                    frame = frame.loc[loc]

                    msg = "Building KD-tree and finding nearest neighbors"
                    logger.debug(msg)
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

                    msg = 'Dissolving and merging network'
                    logger.debug(msg)
                    geometry = shapely.linestrings(COORDS, indices=indices)

                    result['geometry'] = geometry

                    result = (
                        result.frame
                        .pipe(tile2net.grid.pednet.lines.Lines.from_center)
                        .drop2nodes()
                        .frame
                        .set_index('feature')
                        [['geometry']]
                        # .pipe(self.__class__)
                        .pipe(self.__class__.from_frame, wrapper=self)
                    )

                    cache[key] = result

                    msg = (
                        f"Writing {owner.__qualname__}.{self.__name__} "
                        f"to \n\t{file}"
                    )
                    logger.info(msg)
                    result.frame.to_parquet(file)

                    msg = (
                        f'Finished concatenating the network from {len(network)} '
                        f'tiles into a single vector of {len(result)} '
                        f'geometries'
                    )
                    logger.info(msg)
        result.instance = instance
        return result

    locals().update(__get__=_get)

    @property
    def ingrid(self) -> InGrid:
        """Reference to the InGrid instance"""
        return self.instance

    @property
    def feature(self):
        """Feature associated with each line geometry"""
        return self.index.get_level_values('feature')

    @property
    def file(self):
        """
        File at which the network are cached

        Example:
            >>> ingrid: InGrid
            >>> ingrid.network.file
            '/home/<user>/tile2net/ma/network/parquet/Boston Common, MA.parquet'
        """
        return self.ingrid.file.network
        # return self.ingrid.outdir.network.parquet


    def unlink(self):
        """Delete the network file."""
        file = self.file
        msg = (
            f'Uncaching {self.ingrid.__name__}.{self.__name__} and '
            f'deleting file:\n\t{file}'
        )
        logger.info(msg)
        if os.path.exists(file):
            os.remove(file)
        del self.ingrid

    def preview(
            self,
            *args,
            maxdim: int = 2048,
            background: str | tuple[int, int, int] = 'black',
            divider: str = 'grey',
            simplify: float | None = None,
            show: bool = True,
            **kwargs,
    ) -> Image.Image:

        # use the grid's own imagery preview as a basemap
        ingrid = self.ingrid
        mosaic: Image.Image = ingrid.preview(
            maxdim=maxdim,
            divider=divider,
            show=show
        )
        frame = self.frame.to_crs(4326)

        # geometry bounds & scale to determine output pixel size
        minx, miny, maxx, maxy = ingrid.frame.geometry.to_crs(4326).total_bounds
        span_x = max(maxx - minx, 1e-12)
        span_y = max(maxy - miny, 1e-12)
        scale = maxdim / max(span_x, span_y)
        out_w = ceil(span_x * scale)
        out_h = ceil(span_y * scale)
        long_side = max(out_w, out_h)

        # figure canvas
        dpi = 96
        fig_w_in = out_w / dpi
        fig_h_in = out_h / dpi
        fig, ax = plt.subplots(
            figsize=(fig_w_in, fig_h_in),
            dpi=dpi,
            facecolor=background,
        )
        ax.set_facecolor(background)
        # hide axes and ticks; draw full-bleed like polygons preview
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
        ax.margins(0)

        # determine axis/tick color from mosaic luminance for contrast
        # downsample to avoid large reductions
        w, h = mosaic.size
        ds_w = 96
        ds_h = max(1, int(round(h * ds_w / max(1, w))))
        sample = mosaic.resize((ds_w, ds_h), Image.BILINEAR)
        arr = np.asarray(sample).astype(np.float32)
        if arr.ndim == 2:
            lum = float(arr.mean())
        else:
            # Rec. 709 luma
            lum = float(0.2126 * arr[..., 0].mean() +
                        0.7152 * arr[..., 1].mean() +
                        0.0722 * arr[..., 2].mean())
        axis_col = 'white' if lum < 128 else 'black'

        # axes are hidden; no ticks or spines to configure

        # draw the imagery with geographic extent
        # origin='upper' matches the mosaic assembly (row 0 at top)
        img_alpha = float(kwargs.pop('alpha', 1.0))
        ax.imshow(
            mosaic,
            extent=(minx, maxx, miny, maxy),
            origin='upper',
            interpolation='bilinear',
            alpha=img_alpha,
            zorder=1,
        )

        # per-feature line colors and stroke width
        label2color = ingrid.cfg.label2color
        line_w = kwargs.get('width', max(1, long_side // 1400))


        gdf = frame.reset_index()

        # explode multilinestrings → only LineString per row
        geoms = (
            gdf
            .explode(index_parts=False)
            .geometry
        )

        # optional simplification
        if simplify:
            geoms = geoms.simplify(simplify)

        # convert to sequences of xy arrays
        # only keep lines with ≥2 coords
        segments = [
            np.asarray(g.coords)
            for g in geoms
            if g is not None and g.geom_type == "LineString" and len(g.coords) >= 2
        ]

        # feature → color mapping, broadcasted per geometry
        features = gdf.loc[geoms.index, "feature"]
        colors = [
            label2color.get(feat, axis_col)
            for feat in features
        ]

        # make one LineCollection for all features
        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=line_w,
            joinstyle='round',
            capstyle='round',
            zorder=3,
        )

        ax.add_collection(lc)

        # axes bounds, aspect, padding
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format='png',
            facecolor=fig.get_facecolor(),
            bbox_inches=None,
            pad_inches=0.0,
        )
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')

        # optionally pop a live window (don’t close the fig so it stays visible)
        if show:
            try:
                fig.canvas.manager.set_window_title('tile2net — geometry over imagery')
            except Exception:
                pass
            plt.show(block=False)
            plt.pause(0.001)
        else:
            plt.close(fig)

        return pil_img

    def explore(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            simplify=None,
            **kwargs,
    ) -> folium.Map:

        import folium
        feature2color = cfg.label2color
        it = self.frame.groupby('feature', observed=False)
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

    @cached_property
    def disk_usage(self) -> int:
        try:
            return os.path.getsize(self.file)
        except FileNotFoundError:
            return 0

