from __future__ import annotations
from ...grid.frame.namespace import namespace
from ...grid.frame import frame
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

from tile2net.tiles.cfg.logger import logger
from .mask2poly import Mask2Poly
from ..dir.outdir import VecTiles
from ..pednet import PedNet
from ..tiles import file
from ..tiles import tile
from ..tiles.corners import Corners
from ..tiles.tiles import Tiles
from ...tiles.util import recursion_block

if False:
    import folium
    from ..intiles import InTiles


def __get__(
        self: Feature,
        instance: VecTiles,
        owner
) -> Feature:
    self.tiles = instance
    return copy.copy(self)


class Feature(
    namespace,
):
    locals().update(__get__=__get__)
    tiles: VecTiles

    def _ensure_network_column(self, key: str):
        if key not in self.tiles:
            if 'geometry' in self.tiles:
                crs = getattr(self.tiles.geometry, 'crs', None)
            else:
                crs = None
            self.tiles[key] = gpd.GeoSeries(
                [None] * len(self.tiles),
                index=self.tiles.index,
                crs=crs,
                name=key,
            )

    @property
    def polygons(self) -> gpd.GeoSeries:
        key = f'polygons.{self.__name__}'
        if key not in self.tiles:
            self.tiles._load_polygons()
            self._ensure_network_column(key)
        return self.tiles[key]

    @property
    def lines(self) -> gpd.GeoSeries:
        key = f'lines.{self.__name__}'
        if key not in self.tiles:
            self.tiles._load_lines()
            self._ensure_network_column(key)
        return self.tiles[key]


class Tile(
    tile.Tile
):
    tiles: VecTiles

    @tile.cached_property
    def length(self) -> int:
        """
        How many input tiles comprise a segmentation tile.
        This is a multiple of the segmentation tile length.
        """
        vectiles = self.tiles
        intiles = vectiles.intiles
        segtiles = self.segtiles
        result = 2 ** (intiles.tile.scale - self.scale)
        result += 2 * self.padding * segtiles.tile.length
        return result

    @tile.cached_property
    def padding(self) -> int:
        """How many segmentation tiles are used to pad a vector tile"""
        return 1

    @tile.cached_property
    def dimension(self):
        """How many pixels in a segmentation tile"""
        vectiles = self.tiles
        intiles = vectiles.intiles
        result = intiles.tile.dimension * self.length
        return result


class File(
    file.File
):
    tiles: VecTiles

    @property
    def grayscale(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.grayscale'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.vectiles.grayscale.files(tiles)
        if (
                not tiles._stitch_greyscale
                and not files.map(os.path.exists).all()
        ):
            tiles._stitch_greyscale()
        tiles[key] = files
        return tiles[key]

    @property
    def colored(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.mask'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.vectiles.colored.files(tiles)
        if (
                not tiles._stitch_colored
                and not files.map(os.path.exists).all()
        ):
            tiles._stitch_colored()
        tiles[key] = files
        return tiles[key]

    @property
    def infile(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.infile'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.vectiles.infile.files(tiles)
        if (
                not tiles._stitch_infile
                and not files.map(os.path.exists).all()
        ):
            tiles._stitch_infile()
        tiles[key] = files
        return tiles[key]

    @property
    def overlay(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.overlay'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.vectiles.overlay.files(tiles)
        if (
                not tiles._overlay
                and not files.map(os.path.exists).all()
        ):
            tiles._overlay()
        tiles[key] = files
        return tiles[key]

    @property
    def lines(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.lines'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.vectiles.lines.files(tiles)
        if (
                not tiles.vectorize
                and not files.map(os.path.exists).all()
        ):
            tiles.vectorize()
        tiles[key] = files
        return tiles[key]

    @property
    def polygons(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.polygons'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.vectiles.polygons.files(tiles)
        if (
                not tiles.vectorize
                and not files.map(os.path.exists).all()
        ):
            tiles.vectorize()
        tiles[key] = files
        return tiles[key]


def __get__(
        self: VecTiles,
        instance: InTiles,
        owner: type[Tiles],
) -> VecTiles:
    if instance is None:
        return self
    try:
        result: VecTiles = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'intiles.{self.__name__} has not been set. You may '
            f'customize the vectorization functionality by using '
            f'`Intiles.set_vectorization`'
        )
        logger.info(msg)
        cfg = instance.cfg

        scale = cfg.vectile.scale
        length = cfg.vectile.length
        dimension = cfg.vectile.dimension

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
        result = instance.vectiles

    result.tiles = instance
    result.instance = instance

    return result


class VecTiles(
    Tiles
):
    __name__ = 'vectiles'
    locals().update(
        __get__=__get__,
    )

    @tile.cached_property
    def tiles(self) -> InTiles:
        ...

    @File
    def file(self):
        ...

    @property
    def vectiles(self) -> Self:
        return self

    @recursion_block
    def _overlay(self) -> Self:  # noqa: C901
        """
        Create coloured-mask overlays for every vector tile that is still
        missing on disk, using a memory-bounded thread pool.
        """
        vectiles = self

        mask = self.file.colored
        infile = self.file.infile
        overlay = self.file.overlay

        loc = ~overlay.map(os.path.exists)
        pred_files = mask.loc[loc]
        in_files = infile.loc[loc]
        out_files = overlay.loc[loc]

        n_missing = int(np.sum(loc))
        n_total = len(vectiles)

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
            if m.any():
                # round once at the end to preserve precision
                out[m] = np.round(
                    (1.0 - opacity) * base[m].astype(np.float32)
                    + opacity * pred[m].astype(np.float32)
                ).astype(base.dtype)

            iio.imwrite(str(out_path), out)

            # free memory aggressively (huge rasters)
            del base, pred, out, mask_arr, m
            gc.collect()

        # ─────────────────────── scheduler ───────────────────────────── #

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
        intiles = self.segtiles.broadcast
        outtiles = self

        # preemptively predict so logging appears more sequential
        # else you get "now stitching" before "now predicting"
        _ = intiles.file.grayscale

        # only stitch the segtiles which are implicated by the vectiles
        loc = intiles.vectile.xtile.isin(outtiles.xtile)
        loc &= intiles.vectile.ytile.isin(outtiles.ytile)
        intiles = intiles.loc[loc]

        self._stitch(
            small_tiles=intiles,
            big_tiles=outtiles,
            r=intiles.vectile.r,
            c=intiles.vectile.c,
            small_files=intiles.file.grayscale,
            big_files=intiles.vectile.grayscale,
            background=3,
        )

        return self

    @recursion_block
    def _stitch_infile(self) -> Self:
        intiles = self.segtiles.broadcast
        outtiles = self

        # preemptively predict so logging appears more sequential
        # else you get "now stitching" before "now predicting"
        _ = intiles.file.infile

        # only stitch the segtiles which are implicated by the vectiles
        loc = intiles.vectile.xtile.isin(outtiles.xtile)
        loc &= intiles.vectile.ytile.isin(outtiles.ytile)
        intiles = intiles.loc[loc]

        self._stitch(
            small_tiles=intiles,
            big_tiles=outtiles,
            r=intiles.vectile.r,
            c=intiles.vectile.c,
            small_files=intiles.file.infile,
            big_files=intiles.vectile.infile,
        )
        return self

    @property
    def intiles(self):
        return self.tiles

    @recursion_block
    def _stitch_colored(self) -> Self:
        intiles = self.segtiles.broadcast
        outtiles = self

        # preemptively predict so logging appears more sequential
        # else you get "now stitching" before "now predicting"
        _ = intiles.file.colored

        # only stitch the segtiles which are implicated by the vectiles
        loc = intiles.vectile.xtile.isin(outtiles.xtile)
        loc &= intiles.vectile.ytile.isin(outtiles.ytile)
        intiles = intiles.loc[loc]

        self._stitch(
            small_tiles=intiles,
            big_tiles=outtiles,
            r=intiles.vectile.r,
            c=intiles.vectile.c,
            small_files=intiles.file.colored,
            big_files=intiles.vectile.colored,
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
        try:
            if cfg:
                import tile2net.tiles.cfg
                tile2net.tiles.cfg.update(cfg)

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
            # return
            clipped = net.center.clipped.to_crs(4326)
            polys = polys.to_crs(4326)

            polys = (
                polys
                .clip_by_rect(xmin, ymin, xmax, ymax)
                .to_frame('geometry')
                .dissolve(by='feature')
                .explode()
            )

            clipped = (
                clipped
                .clip_by_rect(xmin, ymin, xmax, ymax)
                .to_frame('geometry')
                .dissolve(by='feature')
                .explode()
            )

            Path(polygons_file).parent.mkdir(parents=True, exist_ok=True)
            Path(network_file).parent.mkdir(parents=True, exist_ok=True)
            polys.to_parquet(polygons_file)
            clipped.to_parquet(network_file)

        except Exception as e:
            msg = (
                'Error vectorizing:\n'
                f'affine={affine!r},\n'
                f'infile={infile!r}\n'
                f'{e}'
            )
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
    def vectorize(  # noqa: D401 – no docstrings per user pref
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
            # self._vectorize_submit(*args)
            running[fut] = args[0]  # args[0] == infile path
            return fut

        _ = self.file.grayscale, self.file.colored
        tiles = self

        if not force:
            loc = ~tiles.file.lines.map(os.path.exists)
            loc |= ~tiles.file.polygons.map(os.path.exists)
            tiles = tiles.loc[loc]

        if tiles.empty:
            return
        dest = self.intiles.outdir.vectiles.polygons.dir.rpartition(os.sep)[0]
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

            _cfg = dict(tiles.intiles.cfg)
            it = zip(
                tiles.file.grayscale,
                tiles.affine_params,
                tiles.file.lines,
                tiles.file.polygons,
                tiles.gw,
                tiles.gs,
                tiles.ge,
                tiles.gn,
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
                network_file,  # <- order now matches call-site
                polygons_file,
                xmin,
                ymin,
                xmax,
                ymax
            )
                in it
            )

        total_tiles: int = len(tiles)
        workers: int = os.cpu_count() or 1
        window: int = workers + 1

        tasks = iter(_tile_iter())
        ctx = get_context("spawn")
        running: dict[cf.Future, str] = {}  # Future → infile map

        with self.intiles.cfg, ProcessPoolExecutor(
                mp_context=ctx,
                max_workers=window,
                initializer=self._silence_logging,
                max_tasks_per_child=1,
        ) as pool, tqdm(
            total=total_tiles,
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

        return

    def view(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
    ) -> PIL.Image.Image:

        files = self.file.grayscale
        R: pd.Series = self.r  # 0-based row id
        C: pd.Series = self.c  # 0-based col id

        dim = self.tile.dimension  # original tile side length
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
        tile_h = tile_w  # square tiles
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
            .to_corners(self.segtiles.tile.scale)
            .to_padding()
        )
        return result

    @property
    def affine_params(self) -> pd.Series:
        key = 'affine_params'
        if key in self:
            return self[key]

        dim = self.tile.dimension
        padding = self.padding
        col = 'gw gs ge gn'.split()

        it = padding[col].itertuples(index=False)
        data = [
            rasterio.transform
            .from_bounds(gw, gs, ge, gn, dim, dim)
            for gw, gs, ge, gn in it
        ]
        self[key] = data
        return self[key]

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

    @Tile
    def tile(self):
        ...

# VecTiles.tile.dimension