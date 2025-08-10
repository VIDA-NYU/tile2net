from __future__ import annotations

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
from ..fixed import GeoDataFrameFixed
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
        self.ingrid = instance
        if instance is None:
            result = self
        elif self.__name__ in instance.__dict__:
            result = instance.__dict__[self.__name__]
        else:
            file = self.file
            if os.path.exists(file):
                msg = f"loading {instance.__name__}.{self.__name__} from {file}"
                logger.info(msg)
                result = gpd.read_parquet(file).pipe(self.__class__)
                instance.__dict__[self.__name__] = result
            else:

                msg = f"Stacking geometric columns into a single geometry column."
                logger.debug(msg)
                lines: gpd.GeoDataFrame = instance.vecgrid.lines.copy()
                lines.columns = lines.columns.str.removeprefix('lines.')
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
                    .pipe(self.__class__.from_copy, wrapper=self)
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
                frame = result.loc[iline].reset_index()
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
                    result
                    .pipe(tile2net.grid.pednet.lines.Lines.from_frame)
                    .drop2nodes()
                    .set_index('feature')
                    [['geometry']]
                    .pipe(self.__class__)
                )

                instance.__dict__[self.__name__] = result

                msg = f"Writing {instance.__name__}.{self.__name__} to {file}"
                logger.info(msg)
                result.to_parquet(file)

        result.ingrid = instance
        return result

    __name__ = 'lines'
    ingrid: InGrid = None
    locals().update(
        __get__=_get,
    )

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
            **kwargs,
    ) -> PIL.Image.Image:
        """
        Render the line layer to a static image with latitude/longitude axes.
        """

        # ------------------------------------------------------------------
        # 1. Geometry bounds & scaling
        # ------------------------------------------------------------------
        minx, miny, maxx, maxy = self.geometry.total_bounds
        span_x = max(maxx - minx, 1e-12)
        span_y = max(maxy - miny, 1e-12)

        scale = maxdim / max(span_x, span_y)
        out_w = ceil(span_x * scale)
        out_h = ceil(span_y * scale)
        long_side = max(out_w, out_h)

        # ------------------------------------------------------------------
        # 2. Create a Matplotlib figure sized to the requested raster dims
        # ------------------------------------------------------------------
        dpi = 96  # high-dpi makes large labels sharp without huge font pts
        fig_w_in = out_w / dpi
        fig_h_in = out_h / dpi
        fig, ax = plt.subplots(
            figsize=(fig_w_in, fig_h_in),
            dpi=dpi,
            facecolor=background,
        )
        ax.set_facecolor(background)

        # ------------------------------------------------------------------
        # 3. Choose axis colour that contrasts with the background
        # ------------------------------------------------------------------
        bg_rgb = ImageColor.getrgb(background)
        lum = 0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2]
        axis_col = 'white' if lum < 128 else 'black'

        # Font & tick sizes that scale with the image (in *points*)
        # pts = px / dpi * 72
        labelsize_pt = max(8, int(long_side / dpi * 72 * 0.04))  # 4 % of long side
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

        # ------------------------------------------------------------------
        # 4. Plot each feature line
        # ------------------------------------------------------------------
        label2color = self.ingrid.cfg.label2color
        line_w = kwargs.get('width', max(1, long_side // 1400))

        for feature, series in self.groupby('feature', observed=False).geometry:
            colour = label2color.get(feature, axis_col)
            for geom in series:
                if simplify:
                    geom = geom.simplify(simplify)
                parts = geom.geoms if geom.geom_type == 'MultiLineString' else (geom,)
                for part in parts:
                    coords = np.asarray(part.coords)
                    if coords.shape[0] < 2:
                        continue
                    ax.plot(
                        coords[:, 0],
                        coords[:, 1],
                        color=colour,
                        linewidth=line_w,
                    )

        # ------------------------------------------------------------------
        # 5. Configure the map frame
        # ------------------------------------------------------------------
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')

        # give the axes a little breathing room so labels are never cropped
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.98)

        # ------------------------------------------------------------------
        # 6. Rasterise the figure to a PIL image and return
        # ------------------------------------------------------------------
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format='png',
            facecolor=fig.get_facecolor(),
            bbox_inches=None,  # keep axes & labels
            pad_inches=0.0,
        )
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    def explore(
            self,
            *args,
            grid='cartodbdark_matter',
            m=None,
            simplify=None,
            **kwargs,
    ) -> folium.Map:

        import folium
        feature2color = cfg.label2color
        it = self.groupby('feature', observed=False)
        for feature, frame in it:
            color = feature2color[feature]
            m = explore(
                frame,
                *args,
                color=color,
                name=feature,
                grid=grid,
                simplify=simplify,
                m=m,
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m
