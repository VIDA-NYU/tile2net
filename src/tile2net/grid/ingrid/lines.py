from __future__ import annotations
from .. import util

import io
import os

import PIL.Image
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from PIL import Image, ImageColor
from math import ceil
from scipy.spatial import cKDTree

import tile2net.grid.pednet.lines
from ..cfg import cfg
from ..cfg.logger import logger
from ..explore import explore
from ..frame.framewrapper import FrameWrapper

if False:
    from .ingrid import InGrid
    import folium


class Lines(
    FrameWrapper
):

    def _get(
            self: Lines,
            instance: InGrid,
            owner: type[InGrid]
    ) -> Lines:
        self = super()._get(instance, owner)
        if instance is None:
            result = self
        elif self.__name__ in instance.__dict__:
            result = instance.__dict__[self.__name__]
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
                instance.__dict__[self.__name__] = result
            else:

                msg = f"Stacking geometric columns into a single geometry column."
                logger.debug(msg)
                lines: gpd.GeoDataFrame = instance.vecgrid.lines.frame.copy()
                lines.columns = (
                    lines.columns.str
                    .removeprefix('lines.')
                )
                cols = lines.dtypes == 'geometry'
                result: Lines = (
                    lines
                    .loc[:, cols]
                    .stack(future_stack=True)
                    .rename('geometry')
                    .dropna()
                    .explode()
                    .reset_index()
                    .rename(columns=dict(level_2='feature', ))
                    .pipe(self.__class__.from_frame, wrapper=self)
                )

                instance.__dict__[self.__name__] = result

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

                msg = 'Dissolving and merging lines'
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

                instance.__dict__[self.__name__] = result

                msg = (
                    f"Writing {owner.__qualname__}.{self.__name__} "
                       f"to \n\t{file}"
                )
                logger.info(msg)
                result.frame.to_parquet(file)

                msg = (
                    f'Finished concatenating the lines from {len(lines)} '
                    f'tiles into a single vector of {len(result)} '
                    f'geometries'
                )
                logger.info(msg)
        result.instance = instance
        return result

    __name__ = 'lines'
    locals().update(
        __get__=_get,
    )

    @property
    def ingrid(self) -> InGrid:
        return self.instance

    @property
    def feature(self):
        return self.index.get_level_values('feature')

    @property
    def file(self):
        return self.ingrid.outdir.lines.file

    def unlink(self):
        """Delete the lines file."""
        file = self.file
        msg = (
            f'Uncaching {self.ingrid.__name__}.{self.__name__} and '
            f'deleting file:\n\t{file}'
        )
        logger.info(msg)
        if os.path.exists(file):
            os.remove(file)
        del self.ingrid

    def plot(
            self,
            *args,
            maxdim: int = 2048,
            background: str | tuple[int, int, int] = 'black',
            simplify: float | None = None,
            show: bool = True,
            **kwargs,
    ) -> Image.Image:

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

        # ticks and spines
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

        # plot each linestring (or part of multilinestring) over the basemap
        it = self.frame.groupby('feature', observed=False).geometry
        for feature, series in it:
            colour = label2color.get(feature, axis_col)
            for geom in series:
                g = geom.simplify(simplify) if simplify else geom
                parts = g.geoms if g.geom_type == 'MultiLineString' else (g,)
                for part in parts:
                    coords = np.asarray(part.coords)
                    if coords.shape[0] < 2:
                        continue
                    ax.plot(
                        coords[:, 0],
                        coords[:, 1],
                        color=colour,
                        linewidth=line_w,
                        solid_joinstyle='round',
                        solid_capstyle='round',
                        zorder=3,
                    )

        # axes bounds, aspect, padding
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.98)

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
