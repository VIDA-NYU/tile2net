from __future__ import annotations

import concurrent.futures as cf
import contextlib
import copy
import logging
import os
import os.path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import *

import PIL.Image
import geopandas as gpd
import imageio.v2
import imageio.v3
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
from tile2net.tiles.dir.loader import Loader
from .mask2poly import Mask2Poly
from ..dir.outdir import VecTiles
from ..pednet import PedNet
from ..tiles import file
from ..tiles import tile
from ..tiles.corners import Corners
from ..tiles.tiles import Tiles
from ...tiles.explore import explore
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
    def stitched(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.stitched'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.vectiles.files(tiles)
        if (
                not tiles.stitch
                and not files.map(os.path.exists).all()
        ):
            tiles.stitch()
        tiles[key] = files
        return tiles[key]

    @property
    def lines(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.lines'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.network.files(tiles)
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
            f'VecTiles must be stitched using `SegTiles.stitch` for '
            f'example `SegTiles.stitch.to_dimension(2048)` or '
            f'`SegTiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.intiles = instance
    result.instance = instance

    return result


class VecTiles(
    Tiles
):
    __name__ = 'vectiles'
    locals().update(
        __get__=__get__,
    )

    @Tile
    def tile(self):
        ...

    @File
    def file(self):
        ...

    @property
    def vectiles(self) -> Self:
        return self

    @recursion_block
    def stitch(self) -> Self:
        self.intiles.segtile.xtile
        self.intiles.segtile.ytile
        segtiles = self.segtiles.broadcast
        vectiles = self

        loc = ~segtiles.vectile.stitched.map(os.path.exists)
        infiles = segtiles.file.maskraw.loc[loc]
        row = segtiles.vectile.r.loc[loc]
        col = segtiles.vectile.c.loc[loc]
        group = segtiles.vectile.stitched.loc[loc]

        loc = ~vectiles.file.stitched.map(os.path.exists)
        predfiles = vectiles.file.stitched.loc[loc]
        n_missing = np.sum(loc)
        n_total = len(vectiles)

        if n_missing == 0:  # nothing to do
            msg = f'All {n_total:,} mosaics are already stitched.'
            logger.info(msg)
            return segtiles
        else:
            logger.info(f'Stitching {n_missing:,} of {n_total:,} mosaics missing on disk.')

        loader = Loader(
            files=infiles,
            row=row,
            col=col,
            tile_shape=segtiles.tile.shape,
            mosaic_shape=vectiles.tile.shape,
            group=group
        )

        seen = set()
        for f in predfiles:
            d = Path(f).parent
            if d not in seen:  # avoids extra mkdir syscalls
                d.mkdir(parents=True, exist_ok=True)
                seen.add(d)

        executor = ThreadPoolExecutor()
        imwrite = imageio.v3.imwrite
        it = loader
        it = tqdm(it, 'stitching', n_missing, unit=' mosaic')

        writes = []
        for path, array in it:
            future = executor.submit(imwrite, path, array)
            writes.append(future)
        for w in writes:
            w.result()

        executor.shutdown(wait=True)
        assert vectiles.file.stitched.map(os.path.exists).all()
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

            import tile2net.tiles.cfg
            tile2net.tiles.cfg.update(cfg)
            polys = (
                Mask2Poly
                .from_path(infile, affine)
                .postprocess()
            )

            if polys.empty:
                logging.warning(f'No polygons generated for {infile}')
                return

            net = PedNet.from_polygons(polys)
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
            msg = f'Error vectorizing {infile!r}: {e}'
            raise RuntimeError(msg) from e

    # @recursion_block
    # def vectorize(self):
    #
    #     it = zip(
    #         self.file.stitched,
    #         self.affine_params,
    #         self.file.lines,
    #         self.file.polygons,
    #         self.gw,
    #         self.gs,
    #         self.ge,
    #         self.gn,
    #     )
    #     with (
    #         ProcessPoolExecutor(max_workers=os.cpu_count()) as ex,
    #         self.intiles.cfg,
    #     ):
    #         futures = []
    #         for (
    #                 infile,
    #                 affine,
    #                 network_file,
    #                 polygons_file,
    #                 xmin,
    #                 ymin,
    #                 xmax,
    #                 ymax,
    #         ) in it:
    #             future = ex.submit(
    #                 self._vectorize_submit,
    #                 infile,
    #                 affine,
    #                 xmin,
    #                 ymin,
    #                 xmax,
    #                 ymax,
    #                 polygons_file,
    #                 network_file,
    #             )
    #             futures.append(future)
    #
    #         for fut in futures:
    #             fut.result()
    #

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


    # @recursion_block
    # def vectorize(self) -> None:
    #
    #     iterable = zip(
    #         self.file.stitched,
    #         self.affine_params,
    #         self.file.lines,
    #         self.file.polygons,
    #         self.gw,
    #         self.gs,
    #         self.ge,
    #         self.gn,
    #     )
    #     total = len(self.file.stitched)  # progress bar length
    #
    #     with (
    #         ProcessPoolExecutor(
    #             max_workers=os.cpu_count(),
    #             initializer=self._silence_logging,  # quiet children
    #         ) as ex,
    #         self.intiles.cfg as cfg,
    #     ):
    #         _cfg = dict(cfg)
    #         futures = [
    #             ex.submit(
    #                 self._vectorize_submit,
    #                 infile,
    #                 affine,
    #                 xmin,
    #                 ymin,
    #                 xmax,
    #                 ymax,
    #                 polygons_file,
    #                 network_file,
    #             )
    #             for (
    #                 infile,
    #                 affine,
    #                 network_file,
    #                 polygons_file,
    #                 xmin,
    #                 ymin,
    #                 xmax,
    #                 ymax,
    #             ) in iterable
    #         ]
    #
    #         for fut in tqdm(
    #                 as_completed(futures),
    #                 total=total,
    #                 desc="Vectorizing",
    #                 unit="tile",
    #         ):
    #             fut.result()  # re-raise worker exceptions

    @recursion_block
    def vectorize(  # noqa: D401 – no docstrings per user pref
            self,
    ) -> None:
        """
        Parallel vectorisation with live progress and bounded memory.
        """
        print('⚠️AI GENERATED🤖')

        # ------------------------------------------------------------------ #
        # Build *lazy* iterable ⇢ no up-front list allocation
        # ------------------------------------------------------------------ #
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
            _cfg = dict(self.intiles.cfg)
            it = zip(
                self.file.stitched,
                self.affine_params,
                self.file.lines,
                self.file.polygons,
                self.gw,
                self.gs,
                self.ge,
                self.gn,
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
                for (infile,
                     affine,
                     network_file,  # <- order now matches call-site
                     polygons_file,
                     xmin,
                     ymin,
                     xmax,
                     ymax)
                in it
            )

        total_tiles: int = len(self)
        max_workers: int = os.cpu_count()
        window: int = max_workers * 2  # outstanding futures ≤ 4×cores

        with self.intiles.cfg as cfg_cm, ProcessPoolExecutor(
                max_workers=window,
                initializer=self._silence_logging,       # ← HERE
        ) as pool, tqdm(
                total=total_tiles,
                desc="Vectorizing",
                unit="tile",
                smoothing=0.01,
        ) as pbar:
            _cfg = dict(cfg_cm)  # materialise once, if needed

            it = iter(_tile_iter())  # → yields argument tuples

            # prime the pool
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

        files = self.file.stitched
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
        cols: list[str] = [
            self.sidewalk.lines.name,
            self.crosswalk.lines.name,
            self.road.lines.name,
        ]
        return self[cols]

    @property
    def polygons(self) -> Self:
        cols: list[str] = [
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
            road_color: str = 'green',
            crosswalk_color: str = 'blue',
            sidewalk_color: str = 'red',
            simplify: float = None,
            dash='5, 20',
            attr: str = None,
            **kwargs,
    ) -> folium.Map:
        import folium
        _ = self.road.polygons, self.road.lines

        # m = explore(
        #     self,
        #     geometry='road.polygons',
        #     *args,
        #     color=road_color,
        #     name=f'road.polygons',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     style_kwds=dict(
        #         fill=True,
        #         fillColor=road_color,
        #         fillOpacity=0.05,
        #         weight=0,  # no stroke
        #     ),
        #     highlight=False,
        #     **kwargs,
        # )
        # m = explore(
        #     self,
        #     geometry='sidewalk.polygons',
        #     *args,
        #     color=sidewalk_color,
        #     name=f'sidewalk.polygons',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     style_kwds=dict(
        #         fill=True,
        #         fillColor=sidewalk_color,
        #         fillOpacity=0.05,
        #         weight=0,  # no stroke
        #     ),
        #     highlight=False,
        #     **kwargs,
        # )
        # m = explore(
        #     self,
        #     geometry='crosswalk.polygons',
        #     *args,
        #     color=crosswalk_color,
        #     name=f'crosswalk.polygons',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     style_kwds=dict(
        #         fill=True,
        #         fillColor=crosswalk_color,
        #         fillOpacity=0.05,
        #         weight=0,  # no stroke
        #     ),
        #     highlight=False,
        #     **kwargs,
        # )

        # m = explore(
        #     self,
        #     geometry='road.lines',
        #     *args,
        #     color=road_color,
        #     name=f'road.lines',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     **kwargs,
        # )
        # m = explore(
        #     self,
        #     geometry='sidewalk.lines',
        #     *args,
        #     color=sidewalk_color,
        #     name=f'sidewalk.lines',
        #     tiles=tiles,
        #     simplify=simplify,
        #     m=m,
        #     **kwargs,
        # )

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
            # self,
            # geometry='crosswalk.lines',
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
            # self.geometry.explode(),
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
        # self.crosswalk.lines
        # shapely.get_num_geometries(self)
        return m
