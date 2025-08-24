from __future__ import annotations
from matplotlib.collections import LineCollection


import io
import os
from pathlib import Path

import PIL.Image
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from PIL import Image
from math import ceil
from matplotlib.collections import LineCollection

from tile2net.grid.cfg.logger import logger
from ..benchmark import benchmark
from ..cfg import cfg
from ..explore import explore
from ..frame.framewrapper import FrameWrapper
from ..vecgrid.mask2poly import Mask2Poly

if False:
    from .ingrid import InGrid
    import folium


class Polygons(
    FrameWrapper
):
    __name__ = 'polygons'

    def _get(
            self: Polygons,
            instance: InGrid,
            owner: type[InGrid]
    ) -> Polygons:

        self = super()._get(instance, owner)
        if instance is None:
            result = self
        elif self.__name__ in instance.__dict__:
            result = instance.frame.__dict__[self.__name__]
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

            instance.frame.__dict__[self.__name__] = result

        result.instance = instance

        return result

    locals().update(
        __get__=_get,
    )

    @property
    def ingrid(self) -> InGrid:
        return self.instance

    @property
    def file(self):
        return self.ingrid.outdir.polygons.parquet

    def unlink(self):
        """Delete the polygons file."""
        file = self.file
        msg = (
            f'Uncaching {self.ingrid.__name__}.{self.__name__} and '
            f'deleting file:\n\t{file}'
        )
        logger.info(msg)
        if os.path.exists(file):
            os.remove(file)
        del self.ingrid

    def explore(
            self,
            *args,
            grid='cartodbdark_matter',
            m=None,
            dash='5, 20',
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
        # AI banner
        print('⚠️AI GENERATED🤖')

        # local imports
        from matplotlib.collections import LineCollection

        # z-order per feature
        z_map: dict[str, int] = self.ingrid.cfg.polygon.z_order

        # basemap (must match exactly what preview() produces)
        ingrid = self.ingrid
        mosaic: Image.Image = ingrid.preview(
            maxdim=maxdim,
            divider=divider,
            show=show
        )

        # plot CRS must match the mosaic’s extent CRS (Web-Mercator)
        crs_plot = 3857
        frame = self.frame.to_crs(crs_plot)
        minx, miny, maxx, maxy = (
            ingrid.frame
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
        label2color = self.ingrid.cfg.label2color
        base_width = kwargs.get('width', max(1, long_side // 1400))
        # Use cfg.polygon.thickness if thickness is None
        actual_thickness = thickness if thickness is not None else self.ingrid.cfg.polygon.thickness
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
