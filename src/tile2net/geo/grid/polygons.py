from __future__ import annotations

import io
import os
from functools import cached_property
from pathlib import Path
from typing import *

import PIL.Image
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from PIL import Image
from matplotlib.collections import LineCollection

from tile2net.geo.vecgrid.mask2poly import Mask2Poly
from tile2net.grid.benchmark import benchmark
from tile2net.grid.cfg import cfg
from tile2net.grid.cfg.logger import logger
from tile2net.grid.explore import explore
from tile2net.grid.frame.framewrapper import FrameWrapper

if TYPE_CHECKING:
    from .grid import Grid
    import folium


class Polygons(
    FrameWrapper
):
    """
    Polygons for each feature and region, dissolved across tiles.

    Handles lazy-loading of concatenated polygons from vecgrid tiles:
        >>> Polygons.__get__

    See usage:
        >>> Grid.polygons
    """
    __name__ = 'polygons'

    def __get__(
            self,
            instance: Grid,
            owner: type[Grid]
    ) -> Polygons:
        """
        Lazy-load factory method for accessing polygons for each feature and region, dissolved across tiles.

        Automatically concatenates and dissolves polygon geometries from all
        vec-tiles if not already cached. Results are saved as parquet
        for persistence across sessions.

        Returns:
            Polygons instance with dissolved features from all vecgrid tiles

        Example:
            >>> grid: Grid
            >>> grid.polygons
            Polygons:
                                                    geometry
            feature
            crosswalk  POLYGON ((-7911335.6 5213618.8, -7911339.8 521...
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
            else:
                with instance.polygon_benchmark:
                    vecgrid = instance.vecgrid
                    n = len(instance.vecgrid.polygons)
                    grid_size = max(
                        (affine.a ** 2 + affine.e ** 2)
                        ** .5
                        for affine in vecgrid.affine_params
                    )

                    n_polygons = (
                        vecgrid.polygons.frame
                        .apply(shapely.get_num_geometries)
                        .to_numpy()
                        .sum()
                    )

                    n_features = len(vecgrid.polygons.columns)
                    msg = (
                        f'Aggregating {n_polygons} polygons from {n} tiles and '
                        f'{n_features} features into a single vector'
                    )

                    with benchmark(msg, level='info'):
                        frame = instance.vecgrid.polygons.frame
                        result = (
                            frame
                            .stack(future_stack=True)
                            .to_frame(name='geometry')
                            .set_crs(frame.crs)
                            .pipe(Mask2Poly.from_frame, wrapper=self)
                            .frame
                            .dissolve(by='feature')
                            .explode()
                            .pipe(Polygons.from_frame, wrapper=self)
                        )

                    msg = (
                        f"Writing {owner.__qualname__}.{self.__name__} "
                        f"to \n\t{file}"
                    )
                    logger.info(msg)
                    result.frame.to_parquet(file)

                    msg = (
                        f'Finished concatenating the polygons from {len(instance.vecgrid)} '
                        f'tiles into a single vector of {len(result)} '
                        f'geometries'
                    )
                    logger.info(msg)

            cache[key] = result

        result.instance = instance

        return result

    @property
    def grid(self) -> Grid:
        """Reference to the Ingrid instance."""
        return self.instance

    @property
    def file(self) -> str:
        """
        File at which the polygons are cached.

        Example:
            >>> grid: Grid
            >>> grid.polygons.file
            '/home/<user>/tile2net/ma/polygons/parquet/Boston Common, MA.parquet'

        """
        return self.grid.file.polygons
        # return self.grid.outdir.polygons.parquet

    @property
    def feature(self):
        """Feature associated with each polygon geometry."""
        return self.index.get_level_values('feature')

    def unlink(self):
        """Delete the polygons file."""
        file = self.file
        msg = (
            f'Uncaching {self.grid.__name__}.{self.__name__} and '
            f'deleting file:\n\t{file}'
        )
        logger.info(msg)
        if os.path.exists(file):
            os.remove(file)
        del self.grid

    def explore(
            self,
            *args,
            grid='cartodbdark_matter',
            m=None,
            dash='5, 20',
            simplify=None,
            **kwargs,
    ) -> folium.Map:
        """
        Explore polygons by feature using folium.
        """

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
                grid=grid,
                simplify=simplify,
                style_kwds=dict(
                    fill=False,
                    dashArray=dash,
                ),
                m=m,
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m

    def preview(
            self,
            *args,
            maxdim: int = 2048,
            background: str | tuple[int, int, int] = 'black',
            divider: str = 'grey',
            simplify: float | None = None,
            show: bool = True,
            thickness: float | None = None,
            opacity: float = 1.,
            **kwargs,
    ) -> PIL.Image.Image:
        """ View polygons over imagery as a PIL image. """

        # z-order per feature
        z_map: dict[str, int] = self.grid.cfg.polygon.z_order

        # basemap (must match exactly what preview() produces)
        grid = self.grid
        mosaic: Image.Image = grid.preview(
            maxdim=maxdim,
            divider=divider,
            show=show
        )

        # plot CRS must match the mosaic’s extent CRS (Web-Mercator)
        crs_plot = 3857
        frame = self.frame.to_crs(crs_plot)
        minx, miny, maxx, maxy = (
            grid.frame
            .geometry
            .to_crs(crs_plot)
            .total_bounds
        )

        # figure size locked to mosaic pixels; no rescale
        w_px, h_px = mosaic.size
        dpi = 96
        fig_w_in = w_px / dpi
        fig_h_in = h_px / dpi
        long_side = max(w_px, h_px)

        # figure & axes (full-bleed)
        fig, ax = plt.subplots(
            figsize=(fig_w_in, fig_h_in),
            dpi=dpi,
            facecolor=background,
        )
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
        ax.margins(0)

        # fallback color for unknown features based on luminance
        ds_w = 96
        ds_h = max(1, int(round(h_px * ds_w / max(1, w_px))))
        arr = np.asarray(mosaic.resize((ds_w, ds_h), Image.BILINEAR)).astype(np.float32)
        if arr.ndim == 2:
            lum = float(arr.mean())
        else:
            lum = float(0.2126 * arr[..., 0].mean() + 0.7152 * arr[..., 1].mean() + 0.0722 * arr[..., 2].mean())
        axis_col = 'white' if lum < 128 else 'black'

        # draw mosaic exactly, no padding/border
        base_alpha = float(kwargs.pop('alpha', 1.0))
        ax.imshow(
            mosaic,
            extent=(minx, maxx, miny, maxy),
            origin='upper',
            interpolation='bilinear',
            alpha=max(0.0, min(1.0, base_alpha)),
            zorder=1,
        )

        # stroke setup
        label2color = self.grid.cfg.label2color
        base_width = kwargs.get('width', max(1, long_side // 1400))
        # Use cfg.polygon.thickness if thickness is None
        actual_thickness = thickness if thickness is not None else self.grid.cfg.polygon.thickness
        line_w = max(1, int(round(float(base_width) * float(actual_thickness))))
        lw_hole = max(1, line_w // 2)
        coll_alpha = max(0.0, min(1.0, float(opacity)))

        # vectorized polygon outlines grouped by z-order
        gdf = frame.reset_index()
        geoms = gdf.explode(index_parts=False).geometry
        if simplify:
            geoms = geoms.simplify(simplify, preserve_topology=True)

        mask = (
                geoms.notna()
                & geoms.map(lambda g: getattr(g, 'geom_type', None) == 'Polygon')
                & geoms.map(lambda g: not getattr(g, 'is_empty', True))
        )

        if mask.any():
            feats = gdf.loc[geoms.index, 'feature'][mask]
            colors = feats.map(lambda f: label2color.get(f, axis_col)).to_numpy()
            zvals = feats.map(lambda f: z_map.get(f, 0)).to_numpy()

            layers: dict[int, dict[str, list]] = {}
            polys = geoms[mask].to_numpy()
            for poly, colour, z in zip(polys, colors, zvals):
                bucket = layers.setdefault(z, {'segments': [], 'colors': [], 'widths': []})

                ext = poly.exterior
                if ext is not None:
                    xy = np.asarray(ext.coords)
                    if xy.shape[0] >= 2:
                        bucket['segments'].append(xy)
                        bucket['colors'].append(colour)
                        bucket['widths'].append(line_w)

                if poly.interiors:
                    for ring in poly.interiors:
                        xy = np.asarray(ring.coords)
                        if xy.shape[0] >= 2:
                            bucket['segments'].append(xy)
                            bucket['colors'].append(colour)
                            bucket['widths'].append(lw_hole)

            base_z = 3
            for z in sorted(layers):
                segs = layers[z]['segments']
                if not segs:
                    continue
                lc = LineCollection(
                    segs,
                    colors=layers[z]['colors'],
                    linewidths=layers[z]['widths'],
                    joinstyle='round',
                    capstyle='round',
                    alpha=coll_alpha,
                    zorder=base_z + int(z),
                )
                ax.add_collection(lc)

        # lock bounds & aspect; full-bleed
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # render with zero padding (no border)
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

        if show:
            try:
                fig.canvas.manager.set_window_title('tile2net — polygons over imagery')
            except Exception:
                pass
            plt.show(block=False)
            plt.pause(0.001)
        else:
            plt.close(fig)

        return pil_img

    @cached_property
    def disk_usage(self) -> int:
        try:
            return os.path.getsize(self.file)
        except FileNotFoundError:
            return 0
