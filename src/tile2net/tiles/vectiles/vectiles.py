from __future__ import annotations
from ..tiles import tile

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
import rasterio.features
from PIL import Image
from affine import Affine
from geopandas.array import GeometryDtype
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
from ...tiles.explore import explore
from ...tiles.util import RecursionBlock, recursion_block


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

):
    locals().update(__get__=__get__)
    tiles: VecTiles

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(self, *args, ):
        ...

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
    def indexed(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.prediction'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.vectiles.indexed.files(tiles)
        if (
                not tiles._stitch_prediction
                and not files.map(os.path.exists).all()
        ):
            tiles._stitch_prediction()
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
                not tiles._stitch_mask
                and not files.map(os.path.exists).all()
        ):
            tiles._stitch_mask()
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
        files = tiles.intiles.outdir.lines.files(tiles)
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
        files = tiles.intiles.outdir.polygons.files(tiles)
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


    # @RecursionBlock
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

    # @RecursionBlock
    @recursion_block
    def _stitch_prediction(self) -> Self:
        intiles = self.segtiles.broadcast
        outtiles = self

        # preemptively predict so logging appears more sequential
        # else you get "now stitching" before "now predicting"
        _ = intiles.file.indexed

        self._stitch(
            small_tiles=intiles,
            big_tiles=outtiles,
            r=intiles.vectile.r,
            c=intiles.vectile.c,
            small_files=intiles.file.indexed,
            big_files=intiles.vectile.indexed,
        )

        return self

    @recursion_block
    def _stitch_infile(self) -> Self:
        intiles = self.segtiles.broadcast
        outtiles = self

        # preemptively predict so logging appears more sequential
        # else you get "now stitching" before "now predicting"
        _ = intiles.file.infile

        self._stitch(
            small_tiles=intiles,
            big_tiles=outtiles,
            r=intiles.vectile.r,
            c=intiles.vectile.c,
            small_files=intiles.file.infile,
            big_files=intiles.segtile.infile,
        )
        return self


    @property
    def intiles(self):
        return self.tiles

    # @RecursionBlock
    @recursion_block
    def _stitch_mask(self) -> Self:
        intiles = self.segtiles.broadcast
        outtiles = self

        # preemptively predict so logging appears more sequential
        # else you get "now stitching" before "now predicting"
        _ = intiles.file.colored

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
            polys.to_parquet('polys.parquet')

            if polys.empty:
                logging.warning(f'No polygons generated for {infile}')
                return

            net = PedNet.from_mask2poly(polys)
            net.center.to_parquet('center.parquet')
            clipped = net.center.clipped.to_crs(4326)
            polys = polys.to_crs(4326)

            polys = polys.cx[xmin:xmax, ymin:ymax]
            polys['geometry'] = polys.geometry.clip_by_rect(xmin, ymin, xmax, ymax)
            clipped = clipped.cx[xmin:xmax, ymin:ymax]
            clipped['geometry'] = clipped.geometry.clip_by_rect(xmin, ymin, xmax, ymax)

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
            raise Exception(msg) from e

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

    # @RecursionBlock
    @recursion_block
    def vectorize(  # noqa: D401 – no docstrings per user pref
            self,
    ) -> None:
        """
        Parallel vectorisation with live progress and bounded memory.
        """
        # preemptively stitch so logging appears more sequential
        # else you get "now vectorizing" before "now stitching"
        _ = self.file.indexed, self.file.colored

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
            tiles = self

            _cfg = dict(tiles.intiles.cfg)
            it = zip(
                tiles.file.indexed,
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

        total_tiles: int = len(self)
        max_workers: int = os.cpu_count()
        window: int = max_workers + 1

        with self.intiles.cfg as cfg_cm, ProcessPoolExecutor(
                max_workers=window,
                initializer=self._silence_logging,  # ← HERE
                max_tasks_per_child=1,
        ) as pool, tqdm(
            total=total_tiles,
            # desc="Vectorizing",
            desc=f'{self.__name__}.{self.vectorize.__name__}()',
            unit="tile",
            smoothing=0.01,
        ) as pbar:
            _cfg = dict(cfg_cm)  # materialise once, if needed

            it = iter(_tile_iter())  # → yields argument tuples
            running: set[cf.Future] = {
                pool.submit(self._vectorize_submit, *args)
                for _, args in zip(range(window), it)
            }

            while running:
                # ① wait until at least one worker finishes
                done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)

                # ② handle the finished futures
                for fut in done:
                    running.remove(fut)
                    fut.result()  # re-raise if needed
                    fut.cancel()
                    pbar.update()

                    try:  # ③ refill the pipeline
                        args = next(it)
                    except StopIteration:
                        continue
                    future = pool.submit(self._vectorize_submit, *args)
                    running.add(future)

    def view(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
    ) -> PIL.Image.Image:

        files = self.file.indexed
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

    def _load_lines(self):
        idx_names = list(self.index.names)

        def _read(idx_path):
            idx, path = idx_path
            gdf = gpd.read_parquet(path).reset_index(names='feature')
            if not isinstance(idx, tuple):
                idx = (idx,)
            for name, val in zip(idx_names, idx):
                gdf[name] = val
            gdf['src'] = 'lines'
            return gdf

        tasks = [(i, p) for i, p in self.file.lines.items()]

        with ThreadPoolExecutor() as ex:
            frames = list(ex.map(_read, tasks))

        merged = pd.concat(frames, ignore_index=True, copy=False)

        dissolved = merged.dissolve(
            by=idx_names + ['feature', 'src'],
            as_index=False
        )

        table = (
            dissolved
            .pivot_table(
                values='geometry',
                index=idx_names,
                columns=['feature', 'src'],
                aggfunc='first'
            )
            .reindex(self.index)
        )

        table.columns = [f'lines.{feat}' for feat, _ in table.columns]

        gdf = gpd.GeoDataFrame(table)
        target_crs = getattr(self, 'crs', None)

        for col in gdf.columns:
            if isinstance(gdf[col].dtype, GeometryDtype):
                gdf[col].set_crs(4326, inplace=True, allow_override=True)
                if target_crs and target_crs != 4326:
                    gdf[col] = gdf[col].to_crs(target_crs)

        self[gdf.columns] = gdf
        return self

    def _load_polygons(self):
        idx_names = list(self.index.names)

        def _read(idx_path):
            idx, path = idx_path
            gdf = gpd.read_parquet(path).reset_index(names='feature')
            if not isinstance(idx, tuple):
                idx = (idx,)
            for name, val in zip(idx_names, idx):
                gdf[name] = val
            gdf['src'] = 'polygons'
            return gdf

        tasks = [(i, p) for i, p in self.file.polygons.items()]

        with ThreadPoolExecutor() as ex:
            frames = list(ex.map(_read, tasks))

        merged = pd.concat(frames, ignore_index=True, copy=False)

        dissolved = merged.dissolve(
            by=idx_names + ['feature', 'src'],
            as_index=False
        )

        table = (
            dissolved
            .pivot_table(
                values='geometry',
                index=idx_names,
                columns=['feature', 'src'],
                aggfunc='first'
            )
            .reindex(self.index)
        )

        table.columns = [f'polygon.{feat}' for feat, _ in table.columns]

        gdf = gpd.GeoDataFrame(table)
        target_crs = getattr(self, 'crs', None)

        for col in gdf.columns:
            if isinstance(gdf[col].dtype, GeometryDtype):
                gdf[col].set_crs(4326, inplace=True, allow_override=True)
                if target_crs and target_crs != 4326:
                    gdf[col] = gdf[col].to_crs(target_crs)

        self[gdf.columns] = gdf
        return self

    @Feature
    def road(self):
        ...

    @Feature
    def crosswalk(self):
        ...

    @Feature
    def sidewalk(self):
        ...

    @property
    def lines(self) -> Self:
        cols = [
            self.sidewalk.lines.name,
            self.crosswalk.lines.name,
            self.road.lines.name,
        ]
        return self[cols]

    @property
    def polygons(self) -> Self:
        cols = [
            self.sidewalk.polygons.name,
            self.crosswalk.polygons.name,
            self.road.polygons.name,
        ]
        return self[cols]

    def explore_lines(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            crosswalk_color: str = 'blue',
            sidewalk_color: str = 'red',
            simplify: float = None,
            dash='5, 20',
            **kwargs,
    ) -> folium.Map:
        import folium
        _ = self.road.polygons, self.road.lines

        m = explore(
            self.geometry,
            *args,
            color='grey',
            name=f'tiles',
            tiles=tiles,
            simplify=simplify,
            style_kwds=dict(
                fill=False,
                dashArray=dash,
            ),
            m=m,
            **kwargs,
        )

        m = explore(
            self.crosswalk.lines.explode().rename('geometry'),
            *args,
            color=crosswalk_color,
            name=f'crosswalk.lines',
            tiles=tiles,
            simplify=simplify,
            m=m,
            **kwargs,
        )

        m = explore(
            self.sidewalk.lines.explode().rename('geometry'),
            *args,
            color=sidewalk_color,
            name=f'sidewalk.lines',
            tiles=tiles,
            simplify=simplify,
            m=m,
            **kwargs,
        )

        folium.LayerControl().add_to(m)
        return m

    @Tile
    def tile(self):
        ...

