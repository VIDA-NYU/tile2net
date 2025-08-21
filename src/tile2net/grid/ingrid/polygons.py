from __future__ import annotations

import io
import os

import PIL.Image
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from PIL import Image, ImageColor
from math import ceil

from tile2net.grid.cfg.logger import logger
from .. import util
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
                    result = (
                        instance.vecgrid.polygons.frame
                        .stack(future_stack=True)
                        .set_precision(grid_size=grid_size)
                        .to_frame(name='geometry')
                        .pipe(Mask2Poly.from_frame, wrapper=self)
                        # .postprocess(
                        #
                        # )
                        .frame
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
        return self.ingrid.outdir.polygons.file

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

    def plot(
            self,
            *args,
            maxdim: int = 2048,
            background: str | tuple[int, int, int] = 'black',
            simplify: float | None = None,
            show: bool = True,
            **kwargs,
    ) -> PIL.Image.Image:

        # use the grid's own imagery preview as a basemap
        ingrid = self.ingrid
        mosaic: Image.Image = ingrid.preview(
            maxdim=maxdim,
            divider=None,
            show=show
        )

        # geometry bounds & scale to determine output pixel size
        minx, miny, maxx, maxy = self.geometry.total_bounds
        span_x = max(maxx - minx, 1e-12)
        span_y = max(maxy - miny, 1e-12)
        scale = maxdim / max(span_x, span_y)
        out_w = ceil(span_x * scale)
        out_h = ceil(span_y * scale)
        long_side = max(out_w, out_h)

        # figure canvas sized to raster dims
        dpi = 96
        fig_w_in = out_w / dpi
        fig_h_in = out_h / dpi
        fig, ax = plt.subplots(
            figsize=(fig_w_in, fig_h_in),
            dpi=dpi,
            facecolor=background,
        )
        ax.set_facecolor(background)

        # determine axis/tick color from mosaic luminance for contrast
        w, h = mosaic.size
        ds_w = 96
        ds_h = max(1, int(round(h * ds_w / max(1, w))))
        sample = mosaic.resize((ds_w, ds_h), Image.BILINEAR)
        arr = np.asarray(sample).astype(np.float32)
        if arr.ndim == 2:
            lum = float(arr.mean())
        else:
            # Rec. 709 luma
            lum = float(
                0.2126 * arr[..., 0].mean() +
                0.7152 * arr[..., 1].mean() +
                0.0722 * arr[..., 2].mean()
            )
        axis_col = 'white' if lum < 128 else 'black'

        # ticks and spines sized to image
        labelsize_pt = max(8, int(long_side / dpi * 72 * 0.04))
        ticklen_px = max(4, int(long_side * 0.006))
        ax.tick_params(
            axis='both',
            which='both',
            colors=axis_col,
            direction='out',
            labelsize=labelsize_pt,
            length=ticklen_px,
            width=max(1, ticklen_px // 3),
        )
        for spine in ax.spines.values():
            spine.set_color(axis_col)
            spine.set_linewidth(max(1, ticklen_px // 3))

        # draw the imagery with geographic extent
        img_alpha = float(kwargs.pop('alpha', 1.0))
        ax.imshow(
            mosaic,
            extent=(minx, maxx, miny, maxy),
            origin='upper',
            interpolation='bilinear',
            alpha=img_alpha,
            zorder=1,
        )

        # per-feature polygon edge colors and stroke width
        label2color = self.ingrid.cfg.label2color
        line_w = kwargs.get('width', max(1, long_side // 1400))

        # iterate Polygon/MultiPolygon
        def _iter_polys(g):
            # yields Polygon objects from either Polygon or MultiPolygon
            if g.geom_type == 'Polygon':
                yield g
            elif g.geom_type == 'MultiPolygon':
                yield from g.geoms

        # plot each feature polygon (exterior + holes) over the basemap
        it = self.frame.groupby('feature', observed=False).geometry
        for feature, series in it:
            colour = label2color.get(feature, axis_col)

            for geom in series:
                if simplify:
                    # preserve_topology avoids self-crossing from aggressive simplify
                    geom = geom.simplify(simplify, preserve_topology=True)

                for poly in _iter_polys(geom):
                    if poly.is_empty:
                        continue

                    # exterior ring
                    ext = poly.exterior
                    if ext is not None:
                        ext_xy = np.asarray(ext.coords)
                        if ext_xy.shape[0] >= 2:
                            ax.plot(
                                ext_xy[:, 0],
                                ext_xy[:, 1],
                                color=colour,
                                linewidth=line_w,
                                solid_joinstyle='round',
                                solid_capstyle='round',
                                zorder=3,
                            )

                    # interior rings (holes)
                    for ring in poly.interiors:
                        ring_xy = np.asarray(ring.coords)
                        if ring_xy.shape[0] >= 2:
                            ax.plot(
                                ring_xy[:, 0],
                                ring_xy[:, 1],
                                color=colour,
                                linewidth=max(1, line_w // 2),
                                solid_joinstyle='round',
                                solid_capstyle='round',
                                zorder=3,
                            )

        # configure the map frame
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.98)

        # rasterise the figure to a PIL image
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
                fig.canvas.manager.set_window_title('tile2net — polygons over imagery')
            except Exception:
                pass
            plt.show(block=False)
            plt.pause(0.001)
        else:
            plt.close(fig)

        return pil_img
