from __future__ import annotations
from .. import util
from .. import frame
from .lines import Lines
from .polygons import Polygons
from multiprocessing import get_context

import concurrent.futures as cf
import contextlib
import copy
import gc
import logging
import os
import os.path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import (
    ThreadPoolExecutor,
)
from concurrent.futures import (
    wait,
    FIRST_COMPLETED,
)
from ..grid.corners import Corners
from functools import cached_property
from pathlib import Path
from typing import *
from typing import Iterable

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
from .mask2poly import Mask2Poly
from ..pednet import PedNet
from ...grid.util import recursion_block
from ...grid.frame.namespace import namespace
from ..grid import file

if False:
    import folium
    from ..ingrid import InGrid
from ..grid.grid import Grid


class Feature(
    namespace
):

    def _get(
            self: Feature,
            instance: VecGrid,
            owner
    ) -> Feature:
        self.grid = instance
        return copy.copy(self)

    locals().update(__get__=_get)
    grid: VecGrid

    def _ensure_network_column(self, key: str):
        if key not in self.grid:
            if 'geometry' in self.grid:
                crs = getattr(self.grid.geometry, 'crs', None)
            else:
                crs = None
            self.grid[key] = gpd.GeoSeries(
                [None] * len(self.grid),
                index=self.grid.index,
                crs=crs,
                name=key,
            )

    @property
    def polygons(self) -> gpd.GeoSeries:
        key = f'polygons.{self.__name__}'
        if key not in self.grid:
            self.grid._load_polygons()
            self._ensure_network_column(key)
        return self.grid.frame[key]

    @property
    def lines(self) -> gpd.GeoSeries:
        key = f'lines.{self.__name__}'
        if key not in self.grid:
            self.grid._load_lines()
            self._ensure_network_column(key)
        return self.grid[key]


class File(
    file.File
):
    grid: VecGrid

    @frame.column
    def grayscale(self) -> pd.Series:
        grid = self.grid
        # files = grid.ingrid.outdir.vecgrid.grayscale.files(grid)
        files = grid.ingrid.tempdir.vecgrid.grayscale.files(grid)
        if (
                not grid._stitch_greyscale
                and not files.map(os.path.exists).all()
        ):
            grid._stitch_greyscale()
        return files

    @frame.column
    def colored(self) -> pd.Series:
        grid = self.grid
        # files = grid.ingrid.outdir.vecgrid.colored.files(grid)
        files = grid.ingrid.tempdir.vecgrid.colored.files(grid)
        if (
                not grid._stitch_colored
                and not files.map(os.path.exists).all()
        ):
            grid._stitch_colored()
        return files

    @frame.column
    def infile(self) -> pd.Series:
        grid = self.grid
        # files = grid.ingrid.outdir.vecgrid.infile.files(grid)
        files = grid.ingrid.tempdir.vecgrid.infile.files(grid)
        self.infile = files
        if (
                not grid._stitch_infile
                and not files.map(os.path.exists).all()
        ):
            grid._stitch_infile()
        result = self.infile
        return result

    @frame.column
    def overlay(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.tempdir.vecgrid.overlay.files(grid)
        if (
                not grid._overlay
                and not files.map(os.path.exists).all()
        ):
            grid._overlay()
        return files

    @frame.column
    def lines(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.vecgrid.lines.files(grid)
        if (
                not grid.vectorize
                and not files.map(os.path.exists).all()
        ):
            grid.vectorize()
        return files

    @frame.column
    def polygons(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.vecgrid.polygons.files(grid)
        if (
                not grid.vectorize
                and not files.map(os.path.exists).all()
        ):
            grid.vectorize()
        return files


class VecGrid(Grid):
    __name__ = 'vecgrid'

    def _get(
            self,
            instance: InGrid,
            owner: type[Grid],
    ) -> VecGrid:
        self.instance = instance
        if instance is None:
            return copy.copy(self)
        # cache = instance.__dict__
        # key = self.__name__
        # if instance in cache:
        #     # result: VecGrid = instance.attrs[self.__name__]
        #     result: Self = cache[key]
        #     if result.instance is not instance:
        #         haystack = instance.vectile.index
        #         needles = result.index
        #         loc = needles.isin(haystack)
        #         result = result.loc[loc]
        #
        #         needles = instance.vectile.index
        #         haystack = result.index
        #         loc = needles.isin(haystack)
        #         if not np.all(loc):
        #             msg = 'Not all vectorization tiles implicated by input tiles present'
        #             logger.debug(msg)
        #             del cache[key]
        #             return getattr(instance, self.__name__)

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

            scale = cfg.vector.scale
            length = cfg.vector.length
            dimension = cfg.vector.dimension

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

    @property
    def vecgrid(self) -> Self:
        return self

    @recursion_block
    def _overlay(self) -> Self:  # noqa: C901
        """
        Create coloured-mask overlays for every vector tile that is still
        missing on disk, using a memory-bounded thread pool.
        """
        vecgrid = self

        mask = self.file.colored
        infile = self.file.infile
        overlay = self.file.overlay

        loc = ~overlay.map(os.path.exists)
        pred_files = mask.loc[loc]
        in_files = infile.loc[loc]
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

    @recursion_block
    def _stitch_greyscale(self) -> Self:
        ingrid = self.seggrid.broadcast
        outgrid = self

        # preemptively predict so logging appears more sequential
        # else you get "now stitching" before "now predicting"
        _ = ingrid.file.grayscale

        # only stitch the seggrid which are implicated by the vecgrid
        loc = ingrid.vectile.xtile.isin(outgrid.xtile)
        loc &= ingrid.vectile.ytile.isin(outgrid.ytile)
        # ingrid = ingrid.loc[loc]
        ingrid = (
            ingrid.frame
            .loc[loc]
            .pipe(ingrid.from_frame, wrapper=ingrid)
        )

        self._stitch(
            small_grid=ingrid,
            big_grid=outgrid,
            r=ingrid.vectile.r,
            c=ingrid.vectile.c,
            small_files=ingrid.file.grayscale,
            big_files=ingrid.vectile.grayscale,
            background=3,
        )

        return self

    @recursion_block
    def _stitch_infile(self) -> Self:
        ingrid = self.seggrid.broadcast
        outgrid = self

        # preemptively predict so logging appears more sequential
        # else you get "now stitching" before "now predicting"
        _ = ingrid.file.infile

        # only stitch the seggrid which are implicated by the vecgrid
        loc = ingrid.vectile.xtile.isin(outgrid.xtile)
        loc &= ingrid.vectile.ytile.isin(outgrid.ytile)
        ingrid = ingrid.loc[loc]

        self._stitch(
            small_grid=ingrid,
            big_grid=outgrid,
            r=ingrid.vectile.r,
            c=ingrid.vectile.c,
            small_files=ingrid.file.infile,
            big_files=ingrid.vectile.infile,
        )
        return self

    @property
    def ingrid(self):
        return self.instance

    @recursion_block
    def _stitch_colored(self) -> Self:
        ingrid = self.seggrid.broadcast
        outgrid = self

        # preemptively predict so logging appears more sequential
        # else you get "now stitching" before "now predicting"
        _ = ingrid.file.colored

        # only stitch the seggrid which are implicated by the vecgrid
        loc = ingrid.vectile.xtile.isin(outgrid.xtile)
        loc &= ingrid.vectile.ytile.isin(outgrid.ytile)
        ingrid = ingrid.loc[loc]

        self._stitch(
            small_grid=ingrid,
            big_grid=outgrid,
            r=ingrid.vectile.r,
            c=ingrid.vectile.c,
            small_files=ingrid.file.colored,
            big_files=ingrid.vectile.colored,
        )

        return self

    @staticmethod
    def _vectorize_submit(
            infile: str,
            affine: Affine,
            xmin: float,
            ymin: float,
            xmax: float,
            ymax: float,
            polygons_file: str,
            network_file: str,
            cfg: Optional[dict] = None,
    ):
        polys = None
        Path(polygons_file).parent.mkdir(parents=True, exist_ok=True)
        Path(network_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            if cfg:
                import tile2net.grid.cfg
                tile2net.grid.cfg.update(cfg)

            polys = Mask2Poly.from_path(infile, affine)
            # return

            if polys.empty:
                empty_gdf = (
                    gpd.GeoDataFrame(
                        {"geometry": []},
                        geometry="geometry",
                        crs=4326,
                    )
                    .set_index(
                        pd.Index([], name="feature"),
                    )
                )

                # Persist the empty layers so downstream steps have concrete files
                empty_gdf.to_parquet(polygons_file)
                empty_gdf.to_parquet(network_file)

                msg = f'No polygons generated for {infile}; wrote empty layers instead.'
                logging.warning(msg)
                return

            net = PedNet.from_mask2poly(polys)
            lines = net.center.lines
            edges = lines.edges

            clipped = (
                net.center.clipped
                .frame.to_crs(4326)
                .pipe(
                    net.center.clipped.from_frame,
                    wrapper=net.center.clipped
                )
            )

            polys = (
                polys.frame
                .to_crs(4326)
                .clip_by_rect(xmin, ymin, xmax, ymax)
                .to_frame('geometry')
                .dissolve(by='feature')
                .explode()
            )

            clipped = (
                clipped.frame
                .clip_by_rect(xmin, ymin, xmax, ymax)
                .to_frame('geometry')
                .dissolve(by='feature')
                .explode()
                # .pipe(clipped.from_frame, wrapper=clipped)
            )

            polys.to_parquet(polygons_file)
            clipped.to_parquet(network_file)

        except Exception as e:
            msg = (
                'Error vectorizing:\n'
                f'affine={affine!r},\n'
                f'infile={infile!r}\n'
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
        Disable *all* log records below ERROR **and** swallow bare prints.
        """
        # ❶ Kill log noise (anything < ERROR vanishes)
        logging.disable(logging.ERROR)

        # ❷ Black-hole stdout / stderr
        devnull = open(os.devnull, "w")
        contextlib.redirect_stdout(devnull).__enter__()
        contextlib.redirect_stderr(devnull).__enter__()

    @recursion_block
    def vectorize(
            self,
            force=False,
    ) -> None:
        """
        Parallel vectorisation with live progress and bounded memory.
        """

        # preemptively stitch so logging appears more sequential
        # else you get "now vectorizing" before "now stitching"

        def _submit(
                self,
                executor: ProcessPoolExecutor,
                args: Tuple[
                    str,  # infile
                    Tuple[float, ...],  # affine
                    float,  # xmin
                    float,  # ymin
                    float,  # xmax
                    float,  # ymax
                    str,  # polygons_file
                    str,  # network_file
                    dict,  # cfg
                ],
                running: Dict[cf.Future, str],
        ) -> cf.Future:
            """
            Wrapper that remembers which infile belongs to each Future.
            """
            fut = executor.submit(self._vectorize_submit, *args)
            # fut = self._vectorize_submit(*args)
            running[fut] = args[0]  # args[0] == infile path
            return fut

        _ = self.file.grayscale, self.file.colored
        grid = self

        if not force:
            loc = ~grid.file.lines.map(os.path.exists)
            loc |= ~grid.file.polygons.map(os.path.exists)
            grid = grid.loc[loc]

        if not len(grid):
            return
        dest = (
            self.ingrid.outdir.vecgrid.polygons.dir
            .rpartition(os.sep)
            [0]
        )
        msg = f'Vectorizing to {dest}'
        logger.debug(msg)

        # Build *lazy* iterable ⇢ no up-front list allocation
        def _tile_iter() -> Iterable[
            tuple[
                str,  # infile
                tuple[float, ...],  # affine
                float,  # xmin
                float,  # ymin
                float,  # xmax
                float,  # ymax
                str,  # polygons_file
                str,  # network_file
                dict,  # cfg
            ]
        ]:

            _cfg = dict(grid.ingrid.cfg)

            it = zip(
                grid.file.grayscale,
                grid.affine_params,
                grid.file.lines,
                grid.file.polygons,
                grid.lonmin,
                grid.latmin,
                grid.lonmax,
                grid.latmax,
            )
            yield from (
                (
                    infile,
                    affine,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    polygons_file,
                    network_file,
                    _cfg,
                )
                for (
                infile,
                affine,
                network_file,
                polygons_file,
                xmin,
                ymin,
                xmax,
                ymax
            )
                in it
            )

        total_grid: int = len(grid)
        workers: int = os.cpu_count() or 1
        window: int = workers + 1

        tasks = iter(_tile_iter())
        ctx = get_context("spawn")
        running: dict[cf.Future, str] = {}  # Future → infile map

        with self.ingrid.cfg, ProcessPoolExecutor(
                mp_context=ctx,
                max_workers=window,
                initializer=self._silence_logging,
                max_tasks_per_child=1,
        ) as pool, tqdm(
            total=total_grid,
            desc=f'{self.__name__}.{self.vectorize.__name__}()',
            unit=f' {self.file.lines.name}',
            smoothing=0.01,
        ) as bar:

            # prime the pipeline
            for _, args in zip(range(window), tasks):
                _submit(self, pool, args, running)

            # main loop
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)

                for fut in done:
                    infile = running.pop(fut)
                    try:
                        fut.result()  # ← re-raise worker errors
                    except Exception as exc:
                        # annotate exception with offending file path
                        raise RuntimeError(
                            f'Worker failed while processing {infile!r}:'
                            f'\n\t{exc!r}'
                        ) from exc
                    bar.update()

                    # refill the window
                    try:
                        _submit(self, pool, next(tasks), running)
                    except StopIteration:
                        pass

        msg = f'Finished vectorizing {len(grid)} vectorizing tiles.'
        logger.info(msg)

        if self.cfg.cleanup:
            msg = (
                f'Cleaning up segmentation masks '
                f'from \n\t{self.outdir.seggrid.grayscale.dir} and '
                f'\n\t{self.tempdir.vecgrid.grayscale.dir}'
            )
            logger.info(msg)
            util.cleanup(self.seggrid.file.grayscale)
            util.cleanup(self.vecgrid.file.grayscale)

        return self

    def view(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
    ) -> PIL.Image.Image:

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

    @cached_property
    def padding(self) -> Corners:
        result = (
            self
            .to_corners(self.seggrid.scale)
            .to_padding()
        )
        return result

    @frame.column
    def affine_params(self):

        dim = self.dimension
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
        ...

    @Feature
    def crosswalk(self):
        ...

    @Feature
    def sidewalk(self):
        ...

    @Lines
    def lines(self):
        ...

    @Polygons
    def polygons(self):
        ...

    @cached_property
    def length(self) -> int:
        """
        How many input tiles comprise a segmentation tile.
        This is a multiple of the segmentation tile length.
        """
        vecgrid = self
        ingrid = vecgrid.ingrid
        seggrid = self.seggrid
        result = 2 ** (ingrid.scale - self.scale)
        result += 2 * seggrid.length
        return result
