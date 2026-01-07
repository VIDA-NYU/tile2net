from __future__ import annotations

import concurrent.futures as cf
import copy
import gc
import logging
import os
import os.path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from functools import cached_property
from pathlib import Path
from typing import *

import PIL.Image
import geopandas as gpd
import imageio.v3 as iio
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from PIL import Image
from affine import Affine
from tqdm import tqdm
from tqdm.auto import tqdm

from tile2net.grid.cfg.logger import logger
from .feature import Feature
from .file import File
from .lines import Lines
from .mask2poly import Mask2Poly
from .padded import Padded
from .polygons import Polygons
from .. import frame
from ..cfg.cfg import Cfg
from ..grid.corners import Corners
from ..grid.grid import Grid
from ..loaders.dataset import DataSet
from ..loaders.vec import VecDataSet, VecDataWrapper
from ..pednet import PedNet
from ..sampler.benchmark import Benchmark
from ...grid.util import recursion_block

if False:
    from ..ingrid import InGrid
    from ..seggrid import SegGrid


class VectorizeTask(NamedTuple):
    static: str
    affine: tuple[float, ...]
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    polygons_file: str
    network_file: str
    cfg: Cfg


class VecGrid(Grid):
    """
    "Vectorization Grid" (VecGrid), comprised of "vectorization tiles" (vec-tiles).
    Each vec-tile is a large tile composed of one or more SegGrid tiles, used for
    vectorizing segmentation masks into polygon and line geometries.

    VecGrid tiles are typically larger than SegGrid tiles to enable efficient
    vectorization operations. Each vec-tile covers an area equivalent to multiple
    seg-tiles, minimizing edge artifacts during polygon extraction and centerline
    network generation.

    Example:
        >>> ingrid: InGrid
        >>> ingrid.vecgrid
        VecGrid:
                       lonmin        latmax        lonmax        latmin
        xtile ytile
        9915  12120 -7.911538e+06  5.214840e+06 -7.910315e+06  5.213617e+06

    VecGrid handles:
    - Grouping seg-tiles into larger vec-tiles for vectorization
    - Converting segmentation masks to vector polygons and line geometries
    - Padding tiles to reduce edge artifacts during vectorization

    See usage:
        >>> InGrid.vecgrid

    Handles lazy-loading of VecGrid from InGrid:
        >>> VecGrid._get
    """
    __name__ = 'vecgrid'

    def _get(
            self,
            instance: InGrid,
            owner: type[Grid],
    ) -> VecGrid:
        """
        Lazy-load factory method for accessing VecGrid from InGrid

        Automatically initializes VecGrid using configuration parameters if not already set.
        Uses cached value if available, otherwise calls InGrid.set_vectorization() with
        parameters from cfg.vectorization (scale, length, or dimension).

        Returns:
            VecGrid instance configured for vectorization operations

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid
            VecGrid:
                       lonmin        latmax        lonmax        latmin
            xtile ytile
            9915  12120 -7.911538e+06  5.214840e+06 -7.910315e+06  5.213617e+06
        """
        self.instance = instance
        if instance is None:
            return copy.copy(self)

        cache = instance.frame.__dict__
        key = self.__name__
        if key in cache:
            result = cache[key]

        else:
            msg = (
                f'intiles.{self.__name__} has not been set. You may '
                f'customize the vectorization functionality by using '
                f'`Intiles.set_vectorization`'
            )
            logger.info(msg)
            cfg = instance.cfg

            scale = cfg.vectorization.scale
            length = cfg.vectorization.length
            dimension = cfg.vectorization.dimension

            if scale:
                instance = instance.set_segmentation(scale=scale)
            elif length:
                instance = instance.set_segmentation(length=length)
            elif dimension:
                instance = instance.set_segmentation(dimension=dimension)

            else:
                raise ValueError(
                    'You must set at least one of the following '
                    'segmentation parameters: vectile.scale, vectile.length, or vectile.dimension.'
                )
            result = instance.vecgrid

        result.instance = instance

        return result

    locals().update(__get__=_get)

    @File
    def file(self):
        ...

    @Padded
    def padded(self):
        ...

    @property
    def vecgrid(self) -> Self:
        """
        Reference to VecGrid instance
        """
        return self

    @recursion_block
    def _overlay(self) -> Self:
        """
        Create colorized segmentation overlays on input imagery for visualization

        Alpha-blends colorized segmentation masks onto original input images for each
        vec-tile. Only creates overlays for tiles missing on disk unless
        force=True. Uses memory-bounded threading to avoid RAM exhaustion on large grids.

        Returns:
            Self with overlay files available at vecgrid.file.overlay
        """
        vecgrid = self

        mask = self.file.colorized
        static = self.file.static
        overlay = self.file.overlay

        loc = ~overlay.map(os.path.exists)
        pred_files = mask.loc[loc]
        in_files = static.loc[loc]
        out_files = overlay.loc[loc]

        n_missing = int(np.sum(loc))
        n_total = len(vecgrid)

        if n_missing == 0:
            logger.info(f'All {n_total:,} overlay files on disk.')
            return self

        logger.info(f'{n_missing:,} of {n_total:,} overlay files missing on disk.')

        # ensure output directories exist (avoid redundant mkdir calls)
        seen_dirs: set[Path] = set()
        for f in out_files:
            d = Path(f).parent
            if d not in seen_dirs:
                d.mkdir(parents=True, exist_ok=True)
                seen_dirs.add(d)

        # ───────────────────────── helper ────────────────────────────── #

        def _overlay_single(
                in_path: str | Path,
                pred_path: str | Path,
                out_path: str | Path,
                opacity: float = .20,
        ) -> None:
            """
            Alpha-blend *pred* over *base* wherever *pred* is non-black.

            Parameters
            ----------
            in_path
            pred_path
            out_path
            opacity
                Fraction of *pred* to keep in the blend (0 → invisible,
                1 → fully replace).  Must satisfy 0 ≤ opacity ≤ 1.
            """

            if not 0.0 <= opacity <= 1.0:
                raise ValueError('opacity must be within [0, 1]')

            base = iio.imread(str(in_path))
            pred = iio.imread(str(pred_path))

            # harmonise channel count
            if pred.ndim == 2:  # greyscale → colour
                pred = np.stack((pred,) * base.shape[2], axis=2)
            elif pred.shape[2] > base.shape[2]:  # drop alpha
                pred = pred[..., : base.shape[2]]

            # mask = pixels in pred that are *pure* black
            mask_arr = (
                (pred == 0) if pred.ndim == 2
                else np.all(pred[..., :3] == 0, axis=-1)
            )

            out = base.copy()

            # blend only where pred has content
            m = ~mask_arr
            if np.any(m):
                # round once at the end to preserve precision
                out[m] = np.round(
                    (1.0 - opacity) * base[m].astype(np.float32)
                    + opacity * pred[m].astype(np.float32)
                ).astype(base.dtype)

            iio.imwrite(str(out_path), out)

            # free memory aggressively (huge rasters)
            del base, pred, out, mask_arr, m
            gc.collect()

        def overlay_many(
                *,
                max_inflight: int = 4,  # balance RAM vs. speed
                pool_workers: int | None = None,
        ) -> None:

            triplets = zip(
                in_files,
                pred_files,
                out_files,
                strict=True,  # Py ≥3.12
            )

            pending: set = set()

            with ThreadPoolExecutor(max_workers=pool_workers) as exe, tqdm(
                    total=n_missing,
                    desc='overlay',
                    unit=' img',
                    mininterval=5,
            ) as bar:

                # prime the pipeline
                for _ in range(max_inflight):
                    try:
                        args = next(triplets)
                    except StopIteration:
                        break
                    pending.add(exe.submit(_overlay_single, *args))

                # main loop – always keep ≤ max_inflight futures alive
                while pending:
                    done, pending = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )

                    for fut in done:
                        fut.result()  # re-raise if the task errored
                        bar.update()

                        try:
                            args = next(triplets)
                            pending.add(exe.submit(_overlay_single, *args))
                        except StopIteration:
                            pass

        # pick a sensible default: up to 2-3 GB peak on 64-GB machines
        max_inflight_default = max(2, min(8, os.cpu_count() // 2))
        overlay_many(
            max_inflight=max_inflight_default,
            pool_workers=os.cpu_count(),
        )

        # final sanity check
        assert overlay.map(os.path.exists).all(), 'some overlays are still missing'

        return self

    @staticmethod
    def _vectorize_submit(
            grayscale: np.ndarray,
            affine: Affine,
            xmin: float,
            ymin: float,
            xmax: float,
            ymax: float,
            polygons_file: str,
            network_file: str,
            cfg: Cfg,
    ):
        """
        Worker function for vectorizing a single tile in parallel processing

        Converts a grayscale segmentation mask to vector polygons and extracts
        centerline network. Handles empty masks gracefully by writing empty GeoDataFrames.

        Args:
            grayscale: Segmentation mask as numpy array
            affine: Affine transformation for georeferencing
            xmin: Minimum X coordinate of tile bounds
            ymin: Minimum Y coordinate of tile bounds
            xmax: Maximum X coordinate of tile bounds
            ymax: Maximum Y coordinate of tile bounds
            polygons_file: Output path for polygon parquet file
            network_file: Output path for network lines parquet file
            cfg: Configuration object

        Raises:
            Exception: If vectorization fails for any reason
        """
        polys = None
        Path(polygons_file).parent.mkdir(parents=True, exist_ok=True)
        Path(network_file).parent.mkdir(parents=True, exist_ok=True)
        grid_size = 0.1
        with cfg:
            try:
                # polys = Mask2Poly.from_path(static, affine,crs=3857)
                polys = Mask2Poly.from_array(grayscale, affine, crs=3857)
                assert isinstance(polys, Mask2Poly)
                # return

                if polys.empty:
                    empty_gdf = (
                        gpd.GeoDataFrame(
                            {"geometry": []},
                            geometry="geometry",
                            crs=3857,
                        )
                        .set_index(
                            pd.Index([], name="feature"),
                        )
                    )

                    # Persist the empty layers so downstream steps have concrete files
                    empty_gdf.to_parquet(polygons_file)
                    empty_gdf.to_parquet(network_file)

                    msg = (
                        f'No polygons generated for {polygons_file}; '
                        f'wrote empty layers instead.'
                    )
                    logging.warning(msg)
                    return

                net = (
                    polys
                    .postprocess(
                        min_poly_area=cfg.polygon.min_polygon_area,
                        convexity=cfg.polygon.convexity,
                        simplify=cfg.polygon.simplify,
                        max_hole_area=cfg.polygon.max_hole_area,
                    )
                    .frame.pipe(PedNet.from_mask2poly)
                )

                # precompute lines and edges for clarity
                lines = net.center.lines
                edges = lines.edges

                clipped = (
                    net.center.clipped
                    .frame
                    .pipe(
                        net.center.clipped.from_frame,
                        wrapper=net.center.clipped
                    )
                )

                polys = (
                    net.frame
                    .clip_by_rect(xmin, ymin, xmax, ymax)
                    .to_frame('geometry')
                    .set_precision(grid_size=grid_size)
                    .to_frame('geometry')
                )

                clipped = (
                    clipped.frame
                    .clip_by_rect(xmin, ymin, xmax, ymax)
                    .set_precision(grid_size=grid_size)
                    .to_frame('geometry')
                    .dissolve(by='feature')
                    .explode()
                )

                polys.to_parquet(polygons_file)
                clipped.to_parquet(network_file)

            except Exception as e:
                msg = (
                    'Error vectorizing:\n'
                    f'affine={affine!r},\n'
                )
                if isinstance(polys, gpd.GeoDataFrame):
                    msg += f'polygons_file={polygons_file}\n'
                    polys.to_parquet(polygons_file)
                msg += f'{e}'
                logger.error(msg)
                raise

            finally:
                for name in ("polys", "clipped", "net"):
                    if name in locals():
                        del locals()[name]
                import gc
                gc.collect()

    @staticmethod
    def _silence_logging() -> None:
        """
        Disable log records below ERROR and route stdout/stderr to /dev/null without leaking file descriptors.
        """
        # suppress anything below ERROR
        logging.disable(logging.ERROR)

        # dup stdout/stderr to /dev/null, then close the temporary fd
        fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(fd, 1)
            os.dup2(fd, 2)
        finally:
            os.close(fd)

    @recursion_block
    def vectorize(
            self,
            force=False,
    ) -> None:
        """
        Convert segmentation masks to vector geometries (polygons and lines).

        Processes all vec-tiles in parallel, extracting polygon boundaries and
        centerline networks from semantic segmentation masks. Results are saved as parquet
        files containing GeoDataFrames with geometry for each feature class (sidewalk,
        crosswalk, road).

        Args:
            force: Re-vectorize existing tiles if True, skip existing if False

        Returns:
            None. See output file paths:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.file.polygons
            >>> ingrid.vecgrid.file.lines

        Example:
            >>> ingrid: InGrid = InGrid.from_location('Boston Common, MA')
            >>> ingrid = ingrid.set_source().set_segmentation().set_vectorization()
            >>> ingrid.vecgrid.vectorize()
            Vectorizing to /home/<user>/tile2net/ma/vecgrid/polygons
            vecgrid.vectorize(): 100%|██████| 16/16 [00:45<00:00]
            Finished vectorizing 16 vectorizing tiles.
        """

        dest = (
            self.ingrid.outdir.vecgrid.polygons.dir
            .rpartition(os.sep)
            [0]
        )
        msg = f'Vectorizing to \n\t{dest}'
        logger.debug(msg)

        seggrid: SegGrid = self.seggrid.broadcast

        assert (
            seggrid.file.pred
            .map(os.path.exists)
            .all()
        )

        if not force:
            force = ~seggrid.vectile.line.map(os.path.exists)
            force |= self.cfg.force

        wrapper: VecDataWrapper = VecDataWrapper.from_segtiles(
            static=seggrid.file.pred,
            index=seggrid.vectile.index,
            row=seggrid.vectile.row,
            col=seggrid.vectile.col,
            background=3,
            force=force,
            affine=seggrid.vectile.affine,
            lonmin=seggrid.vectile.lonmin,
            latmin=seggrid.vectile.latmin,
            lonmax=seggrid.vectile.lonmax,
            latmax=seggrid.vectile.latmax,
            polygon_file=seggrid.vectile.polygon_file,
            line_file=seggrid.vectile.line_file,
        )

        total = wrapper.index.nunique()
        if not total:
            msg = 'All vector tiles already exist on disk.'
            logger.info(msg)
            return
        threads = self.cfg.vectorization.num_loaders
        dataset = VecDataSet(wrapper, threads=threads)
        loader = dataset.loader

        bar = tqdm(
            total=total,
            desc=f'vecgrid.{self.vectorize.__name__}()',
            unit=f' {self.file.lines.name}',
            smoothing=0.01,
            mininterval=10,
        )

        with self.ingrid.cfg, bar, self.benchmark :
            for minibatch in loader:
                bar.update(len(minibatch))

        msg = f'Finished vectorizing {total} vectorizing tiles.'
        logger.info(msg)

    def view(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
    ) -> PIL.Image.Image:
        """
        Generate a mosaic preview of all vec-tiles in the grid.

        Creates a single image showing all vec-tiles arranged in their spatial grid pattern,
        with optional divider lines between tiles. Automatically downscales to fit within maxdim.

        Args:
            maxdim: Maximum dimension for the output image in pixels
            divider: Color name for tile divider lines (None for no divider)

        Returns:
            PIL Image containing the tile mosaic

        Example:
            >>> ingrid: InGrid
            >>> img = ingrid.vecgrid.view(maxdim=4096, divider='red')
            >>> img.show()
        """

        files = self.file.grayscale
        R: pd.Series = self.r  # 0-based row id
        C: pd.Series = self.c  # 0-based col id

        dim = self.dimension  # original tile side length
        n_rows = int(R.max()) + 1
        n_cols = int(C.max()) + 1
        div_px = 1 if divider else 0

        # full mosaic size before optional down-scaling
        full_w0 = n_cols * dim + div_px * (n_cols - 1)
        full_h0 = n_rows * dim + div_px * (n_rows - 1)

        scale = 1.0
        if max(full_w0, full_h0) > maxdim:
            scale = maxdim / max(full_w0, full_h0)

        tile_w = max(1, int(round(dim * scale)))
        tile_h = tile_w  # square grid
        full_w = n_cols * tile_w + div_px * (n_cols - 1)
        full_h = n_rows * tile_h + div_px * (n_rows - 1)

        canvas_col = divider if divider else (0, 0, 0)
        mosaic = Image.new('RGB', (full_w, full_h), color=canvas_col)

        def load(idx: int) -> tuple[int, int, np.ndarray]:
            arr = iio.imread(files.iat[idx])
            if scale != 1.0:
                arr = np.asarray(
                    Image.fromarray(arr).resize(
                        (tile_w, tile_h), Image.Resampling.LANCZOS
                    )
                )
            return R.iat[idx], C.iat[idx], arr

        with ThreadPoolExecutor() as pool:
            for r, c, arr in pool.map(load, range(len(files))):
                x0 = c * (tile_w + div_px)
                y0 = r * (tile_h + div_px)
                mosaic.paste(Image.fromarray(arr), (x0, y0))

        return mosaic

    def _iter_tasks(
            self,
            grid: VecGrid,
            loader: DataSet,
            cfg: Cfg,
    ) -> Iterator[VectorizeTask]:
        it = zip(
            loader,
            grid.affine_params,
            grid.lonmin,
            grid.latmin,
            grid.lonmax,
            grid.latmax,
            grid.file.polygons,
            grid.file.lines,
        )
        for (
                static,
                affine,
                xmin,
                ymin,
                xmax,
                ymax,
                polygons_file,
                network_file,
        ) in it:
            yield VectorizeTask(
                static=static,
                affine=affine,
                xmin=float(xmin),
                ymin=float(ymin),
                xmax=float(xmax),
                ymax=float(ymax),
                polygons_file=polygons_file,
                network_file=network_file,
                cfg=cfg,
            )

    def _bounded_as_completed(
            self,
            executor: ProcessPoolExecutor,
            tasks: Iterator[VectorizeTask],
            max_inflight: int,
    ) -> Iterator[tuple[cf.Future, VectorizeTask]]:
        inflight: dict[cf.Future, VectorizeTask] = {}

        # prefill up to the window size
        for _ in range(max_inflight):
            try:
                task = next(tasks)
            except StopIteration:
                break
            fut = executor.submit(self._vectorize_submit, *task)
            inflight[fut] = task

        # main loop
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                task = inflight.pop(fut)
                yield fut, task
                try:
                    nxt = next(tasks)
                except StopIteration:
                    continue
                new_fut = executor.submit(self._vectorize_submit, *nxt)
                inflight[new_fut] = nxt

    @cached_property
    def padding(self) -> Corners:
        """
        Corner coordinates for padded vec-tiles.

        Computes the tile corners after applying padding, used for proper
        alignment during vectorization to avoid edge artifacts.

        Returns:
            Corners object with xmin, ymin, xmax, ymax for each padded tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.padding
            Corners with expanded bounds for padding
        """
        result = (
            self
            .to_corners(self.seggrid.scale)
            .to_padding()
        )
        return result

    @frame.column
    def affine_params(self):
        """
        Affine transformation parameters for georeferencing each tile.

        Computes the affine transformation matrix that maps pixel coordinates
        to geographic coordinates (EPSG:3857) for each vec-tile.

        Returns:
            Series of Affine objects, one per tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.affine_params
            xtile  ytile
            9915   12120    Affine(0.298..., 0.0, -7911538.18..., ...)
        """
        dim = self.vecgrid.padded.dimension
        padding = self.padding
        col = [
            self.lonmin.name,
            self.latmin.name,
            self.lonmax.name,
            self.latmax.name,
        ]

        it = (
            padding.frame
            [col]
            .itertuples(index=False)
        )

        result = [
            rasterio.transform
            .from_bounds(gw, gs, ge, gn, dim, dim)
            for gw, gs, ge, gn in it
        ]
        return result

    @Feature
    def road(self):
        # todo: is this still needed?
        raise NotImplementedError

    @Feature
    def crosswalk(self):
        # todo: is this still needed?
        raise NotImplementedError

    @Feature
    def sidewalk(self):
        # todo: is this still needed?
        raise NotImplementedError

    @Lines
    def lines(self):
        """
        Dissolved line features from all vec-tiles.

        Returns MultiLineString geometries for each feature (sidewalk, crosswalk)
        organized by tile coordinates.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.lines
            Lines:
            feature                                              crosswalk
            xtile ytile
            9915  12120  MULTILINESTRING ((-7910926 5213692.6, -7910925...
            feature                                               sidewalk
            xtile ytile
            9915  12120  MULTILINESTRING ((-7910947.3 5213616.8, -79109...
        """

    @Polygons
    def polygons(self):
        """
        Dissolved polygon features from all vec-tiles.

        Returns MultiPolygon geometries for each feature (sidewalk, crosswalk, road)
        organized by tile coordinates.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.polygons
            Polygons:
            feature                                              crosswalk
            xtile ytile
            9915  12120  MULTIPOLYGON (((-7910483.6 5213928.8, -7910483...
            feature                                                   road
            xtile ytile
            9915  12120  MULTIPOLYGON (((-7910857.1 5214839.8, -7910844...
            feature                                               sidewalk
            xtile ytile
            9915  12120  MULTIPOLYGON (((-7910400.7 5214764, -7910400.4...
        """

    @cached_property
    def length(self) -> int:
        """
        Number of SegGrid tiles that comprise one dimension of a vec-tile.

        For example, if SegGrid uses zoom 18 and VecGrid uses zoom 16,
        each VecGrid tile is 2^2 = 4 SegGrid tiles wide.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.length
            4
        """
        result = 2 ** (self.seggrid.scale - self.scale)
        return result

    @cached_property
    def benchmark(self):
        """
        Benchmark for vectorization operations (RAM, CPU).
        """
        result = Benchmark(include_gpu=False)
        return result

    @cached_property
    def dimension(self) -> int:
        """
        Pixel dimension of each vec-tile.

        For example, if seg-tiles are 1024x1024 pixels and length is 4,
        vec-tiles are 4096x4096 pixels.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.dimension
            4096
        """
        return self.seggrid.dimension * self.length

    @property
    def ingrid(self) -> InGrid:
        """
        Reference to the parent InGrid instance.

        Returns:
            InGrid instance that this VecGrid belongs to

        Example:
            >>> ingrid: InGrid
            >>> vecgrid = ingrid.vecgrid
            >>> vecgrid.ingrid is ingrid
            True
        """
        return self.instance
