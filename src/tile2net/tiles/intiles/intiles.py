from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ...raster.tile_utils.topology import extend_line

# thread-local store
tls = threading.local()

import os
import os.path
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import *

import PIL.Image
import certifi
import geopandas as gpd
import imageio.v3 as iio
import numpy as np
import pandas as pd
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from tile2net.raster import util
from tile2net.tiles.cfg import cfg
from tile2net.tiles.cfg.logger import logger
from tile2net.tiles.dir.dir import Dir, ExtensionNotFoundError, XYNotFoundError
from tile2net.tiles.dir.indir import Indir
from tile2net.tiles.dir.outdir import Outdir
from . import delayed
from .lines import Lines
from .polygons import Polygons
from .segtile import SegTile
from .vectile import VecTile
from .source import Source, SourceNotFound
from .. import util
from ..cfg import Cfg
from ..segtiles import SegTiles
from ..tiles import tile, file
from ..tiles.tiles import Tiles
from ..util import assert_perfect_overlap
from ..vectiles.vectiles import VecTiles
from ...tiles.util import RecursionBlock, recursion_block

if False:
    from .padded import Padded


class Tile(
    tile.Tile
):
    tiles: InTiles

    @tile.cached_property
    def zoom(self) -> int:
        """
        The zoom level of the tile.
        """
        return self.tiles.tile.scale

    @tile.cached_property
    def length(self):
        return 1

    @tile.cached_property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        try:
            sample = next(
                p
                for p in self.tiles.file.infile
                if Path(p).is_file()
            )
        except StopIteration:
            raise FileNotFoundError('No image files found to infer dimension.')
        return iio.imread(sample).shape[1]  # width


class File(
    file.File
):
    tiles: InTiles

    @property
    def infile(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.infile'
        if key in tiles:
            return tiles[key]
        files = tiles.indir.files(tiles)
        tiles[key] = files
        if (
                not tiles.download
                and not files.map(os.path.exists).all()
        ):
            tiles.download()
        return tiles[key]


class InTiles(
    Tiles
):
    __name__ = 'intiles'

    @VecTiles
    def vectiles(self):
        """
        After performing SegTiles.stitch, SegTiles.vectiles is
        available for performing inference on the stitched tiles.
        """

    @SegTiles
    def segtiles(self):
        """
        After performing InTiles.stitch, InTiles.segtiles is
        available for performing inference on the stitched tiles.
        """

    @Source
    def source(self):
        """
        Returns the Source class, which wraps a tile server.
        See `Tiles.with_source()` to actually set a source.
        """
        # This code block is just semantic sugar and does not run.
        # These methods are how to set the source:
        intiles = self.set_source(...)  # automatically sets the source
        intiles = self.set_source('nyc')

    # @Indir
    # def indir(self):
    #     """
    #     Returns the Indir class, which wraps an input directory, for
    #     example `input/dir/x_y_z.png` or `input/dir/x/y/z.png`.
    #     See `Tiles.with_indir()` to actually set an input directory.
    #     """
    #     # This code block is just semantic sugar and does not run.
    #     # This method is how to set the input directory:
    #     self.set_indir(...)

    @Indir
    def indir(self):
        intiles = self.intiles.outdir.intiles
        extension = self.source.extension
        format = os.path.join(
            intiles.dir,
            'infile',
            # f'z/x_y.{extension}'
            f'z/x_y'
        )
        result = Indir.from_format(format)
        return result

    # @property
    # def indir(self):
    #     return self.outdir.intiles.infile
    #
    # @indir.setter
    # def indir(self, indir: Union[str, Path]):
    #     """
    #     Set the input directory for the tiles.
    #     """
    #     self.outdir.intiles.infile = indir

    @Outdir
    def outdir(self):
        ...

    @Lines
    def lines(self):
        ...

    @Polygons
    def polygons(self):
        ...

    @classmethod
    def from_location(
            cls,
            location,
            zoom: int = None,
    ) -> Self:
        latlon = util.geocode(location)
        result = cls.from_bounds(
            latlon=latlon,
            zoom=zoom
        )
        return result

    @classmethod
    def from_bounds(
            cls,
            latlon: Union[
                str,
                list[float],
            ],
            zoom: int = None,
    ) -> Self:
        """
        Create some base Tiles from bounds in lat, lon format and a
        slippy tile zoom level.
        """
        if zoom is None:
            zoom = cfg.zoom
        if isinstance(latlon, str):
            if ', ' in latlon:
                split = ', '
            elif ',' in latlon:
                split = ','
            elif ' ' in latlon:
                split = ' '
            else:
                raise ValueError(
                    "latlon must be a string with coordinates "
                    "separated by either ', ', ',', or ' '."
                )
            latlon = [
                float(x)
                for x in latlon.split(split)
            ]
        gn, gw, gs, ge = latlon
        gn, gs = min(gn, gs), max(gn, gs)
        gw, ge = min(gw, ge), max(gw, ge)
        tw, tn = util.deg2num(gw, gn, zoom=zoom)
        te, ts = util.deg2num(ge, gs, zoom=zoom)
        te, tw = max(te, tw), min(te, tw)
        ts, tn = max(ts, tn), min(ts, tn)
        te += 1
        ts += 1
        tx = np.arange(tw, te)
        ty = np.arange(tn, ts)
        index = pd.MultiIndex.from_product([tx, ty])
        tx = index.get_level_values(0)
        ty = index.get_level_values(1)
        result = cls.from_integers(tx, ty, scale=zoom)
        return result

    # @classmethod
    # def from_config(
    #         cls,
    #         args
    # ):
    #     from tile2net.tiles.cfg import Cfg
    #     cfg = vars(args)
    #     cfg = Cfg(cfg)
    #
    #     tiles = Tiles.from_location(
    #         location=cfg.location,
    #         zoom=cfg.zoom,
    #     )
    #
    #     if cfg.source:
    #         # use specified source
    #         tiles = tiles.set_source(
    #             source=cfg.source,
    #             indir=cfg.input_dir,
    #         )
    #     elif cfg.input_dir:
    #         # use local files
    #         tiles = tiles.set_indir(
    #             indir=cfg.input_dir,
    #         )
    #     else:
    #         # infer source from location
    #         tiles = tiles.set_source(
    #             indir=cfg.input_dir,
    #         )
    #
    #     pad = cfg.padding
    #     stitch = tiles.stitch
    #     if cfg.segtile.dimension:
    #         tiles = stitch.to_dimension(cfg.segtile.dimension, pad)
    #     elif cfg.segtile.mosaic:
    #         tiles = stitch.to_mosaic(cfg.segtile.mosaic, pad)
    #     elif cfg.segtile.scale:
    #         tiles = stitch.to_scale(cfg.segtile.scale, pad)
    #
    #     tiles.predict(cfg.output_dir)
    #     raise NotImplemented

    @Cfg
    def cfg(self):
        # This code block is just semantic sugar and does not run.
        # You can access the various configuration options this way:
        _ = self.cfg.zoom
        _ = self.cfg.model.bs_val
        _ = self.cfg.polygon.max_hole_area
        # Please do not set the configuration options directly,
        # you may introduce bugs.

    @SegTile
    def segtile(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the tiles have been stitched:
        _ = (
            # xtile of the larger mosaic
            self.segtile.xtile,
            # ytile of the larger mosaic
            self.segtile.ytile,
            # row of the tile within the larger mosaic
            self.segtile.r,
            # column of the tile within the larger mosaic
            self.segtile.c,
        )

    @VecTile
    def vectile(self):
        # This code block is just semantic sugar and does not run.
        _ = (
            # xtile of the larger mosaic
            self.segtile.xtile,
            # ytile of the larger mosaic
            self.segtile.ytile,
        )


    @recursion_block
    def download(
            self,
            retry: bool = True,
            force: bool = False,
            max_workers: int = 64,  # ⇣ lower default ⇣ avoids port-exhaustion
    ) -> Self:

        if self is not self.padded:
            return self.padded.download(
                retry=retry,
                force=force,
                max_workers=max_workers,
            )

        paths = self.file.infile
        urls = self.source.urls
        if paths.empty or urls.empty:
            return self
        if len(paths) != len(urls):
            raise ValueError("paths and urls must be equal length")

        # ── ensure directories exist
        for p in {Path(p).parent for p in paths}:
            p.mkdir(parents=True, exist_ok=True)

        exists = paths.map(os.path.exists).to_numpy(bool)
        if exists.all() and not force:
            logger.info("All tiles already on disk.")
            return self

        if not force:
            paths = paths[~exists]
            urls = urls[~exists]

        mapping = dict(zip(paths.array, urls.array))
        if not mapping:
            return self

        msg = f"Downloading {len(mapping):,} tiles to {self.indir.dir}"
        logger.info(msg)

        pool_size = min(max_workers, len(mapping))
        not_found: list[Path] = []
        failed: list[Path] = []

        # ── helper that both downloads *and* writes inside the same thread
        def _fetch_write(
                path: Path,
                url: str,
        ) -> None:
            resp = tls.session.get(url, stream=True, timeout=(3, 30))
            status = resp.status_code
            if status == 404:
                not_found.append(path)
                resp.close()
                return
            if status != 200:
                failed.append(path)
                resp.close()
                return

            # atomic write
            tmp = Path(
                tempfile.mkstemp(
                    dir=path.parent,
                    suffix=".part",
                )[1]
            )
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 15):
                    fh.write(chunk)
            resp.close()

            # quick sanity check + move
            try:
                iio.imread(tmp)
            except ValueError:
                tmp.unlink(missing_ok=True)
                failed.append(path)
                return
            shutil.move(tmp, path)

        # ── threaded execution
        with ThreadPoolExecutor(
                max_workers=pool_size,
                initializer=self._init_net_worker,  # one Session per thread
        ) as pool:

            # submit *all* jobs up front (cheap) …
            futures = {
                pool.submit(_fetch_write, Path(p), u)
                for p, u in mapping.items()
            }

            # … and drive a single tqdm on completion
            for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Downloading",
                    unit="tile",
            ):
                fut.result()

        if len(not_found) + len(failed) == len(self.file.infile):
            msg = f'All {len(not_found) + len(failed):,} tiles failed to download.'
            raise FileNotFoundError(msg)

        if not_found:
            # black = self.static.black
            # black = getattr(self.static.black, self.)
            try:
                black = getattr(self.static.black, self.source.extension)
            # except AttributeError as e:
            except AttributeError as e:
                msg = f'No black placeholder found for extension {self.source.extension}.'

                raise FileNotFoundError(msg) from e
            logger.warning("%d tiles returned 404 – linking placeholder.", len(not_found))
            for p in not_found:
                try:
                    os.symlink(black, p)
                except (OSError, NotImplementedError):
                    shutil.copy2(black, p)

        if failed:
            if retry:
                logger.error("%d failed; retrying once.", len(failed))
                return self.download(
                    retry=False,
                    force=False,
                    max_workers=max_workers,
                )
            raise FileNotFoundError(f"{len(failed):,} tiles failed.")

        logger.info("All requested tiles are on disk.")
        return self

    def set_source(
            self,
            source=None,
            outdir: Union[str, Path] = None,
    ) -> Self:
        """
        Assign a source to the tiles. The tiles are downloaded from
        the source and saved to an input directory.

        You can pass an outdir which will be the destination the files
        are saved to.
        """
        if source is None:
            source = (
                gpd.GeoSeries([self.union_all()], crs=self.crs)
                .to_crs(4326)
                .pipe(Source.from_inferred)
            )
        elif isinstance(source, Source):
            source = copy.copy(source)
        else:
            try:
                source = Source.from_inferred(source)
            except SourceNotFound as e:
                msg = (
                    f'Unable to infer a source from {source}. '
                    f'Please specify a valid source or use a '
                    f'different method to set the tiles.'
                )
                raise ValueError(msg) from e

        result = self.copy()
        result.source = source
        msg = (
            f'Setting source to {source.__class__.__name__} '
            f'({source.name})'
        )
        logger.info(msg)

        if not outdir:
            outdir = (
                Path(f'./{source.name}')
                .expanduser()
                .resolve()
                .__str__()
            )

        try:
            dir = Dir.from_format(outdir)
        except ExtensionNotFoundError:
            # dir = f'{outdir}.{source.extension}'
            dir =  outdir
            try:
                dir = Dir.from_format(dir)
            except XYNotFoundError as e:
                # dir = f'{outdir}/z/x_y.{source.extension}'
                dir = f'{outdir}/z/x_y'
                dir = Dir.from_format(dir)

        except XYNotFoundError as e:
            # msg = (
            #     f'Invalid output directory: extension included but '
            #     f'no `x` or `y` in the format: {outdir}. '
            # )
            # raise ValueError(msg) from e
            dir = f'{outdir}/z/x_y'
            dir = Dir.from_format(dir)


        # if result.source.extension != dir.extension:
        #     msg = (
        #         f'Output directory {outdir} has a different '
        #         f'extension ({dir.extension}) than the source '
        #         f'({result.source.extension}).'
        #     )
        #     raise ValueError(msg)
        outdir = dir.original

        result = result.set_outdir(outdir)

        # infile = result.outdir.intiles.infile
        # indir = infile.format.replace(infile.extension, source.extension)
        # result = result.set_indir(indir)

        return result

    def set_indir(
            self,
            indir: str = None,
            name: str = None,
    ) -> Self:
        """
        Assign an input directory to the tiles. The directory must
        implicate the X and Y tile numbers in the file names.

        Good:
            input/dir/x/y/z.png
            input/dir/x/y/.png
            input/dir/x_y_z.png
            input/dir/x_y.png
            input/dir/z/x_y.png
        Bad:
            input/dir/x.png
            input/dir/y

        This will set the input directory and format
        self.with_indir('input/dir/x/y/z.png')

        There is no format specified to, so it will default to
        input/dir/x_y_z.png:
        self.with_indir('input/dir')

        This will fail to explicitly set the format, so it will
        default to input/dir/x/x_y_z.png
        self.with_indir('input/dir/x')
        """
        result = self.copy()
        if name:
            result.name = name
        try:
            result.indir = indir
            # result.outdir.intiles.infile = indir
            indir: Indir = result.indir
            msg = f'Setting input directory to \n\t{indir.original}. '

            logger.info(msg)

        except ValueError as e:
            msg = (
                f'Invalid input directory: {indir}. '
                f'The directory directory must implicate the X and Y '
                f'tile numbers by including `x` and `y` in some format, '
                f'for example: '
                f'input/dir/x/y/z.png or input/dir/x_y_z.png.'
            )
            raise ValueError(msg) from e
        # try:
        #     _ = result.outdir
        # except AttributeError:
        #     msg = (
        #         f'Output directory not yet set. Based on the input directory, '
        #         f'setting it to a default value.'
        #     )
        #     logger.info(msg)
        #     result = result.set_outdir()
        return result

    def set_outdir(
            self,
            outdir: Union[str, Path] = None,
    ) -> Self:
        """
        Assign an output directory to the tiles.
        The tiles are saved to the output directory.
        """
        result: Tiles = self.copy()

        # if not outdir:
        # if not outdir:
        #     if cfg.output_dir:
        #         outdi = cfg.output_dir

        if not outdir:
            outdir = cfg.output

        if not outdir:
            raise ValueError(f'No output directory specified. ')

        dir = outdir
        try:
            result.outdir = dir
        except XYNotFoundError as e:
            _dir = f'{outdir}/z/x_y'
            msg = (
                f'XYZ format not found in output directory: {dir}. '
                f'Defaulting to {_dir}'
            )
            logger.info(msg)
            dir = _dir

        try:
            result.outdir = dir
        except ExtensionNotFoundError:
            dir = f'{dir}.png'

        result.outdir = dir
        # noinspection PyUnresolvedReferences
        msg = f'Setting output directory to \n\t{result.outdir.original} '
        logger.info(msg)

        return result

    def set_segmentation(
            self,
            *,
            dimension: int = None,
            length: int = None,
            mosaic: int = None,
            scale: int = None,
            fill: bool = True,
            bs_val: int = None,
    ) -> Self:
        from ..segtiles import SegTiles
        # todo: if all are None, determine dimension using VRAM
        scale = self._to_scale(dimension, length, mosaic, scale)

        if bs_val:
            self.cfg.model.bs_val = bs_val

        msg = 'Padding InTiles to align with SegTiles'
        logger.debug(msg)
        intiles = (
            self
            .to_scale(scale, fill=fill)
            .to_scale(self.tile.scale, fill=fill)
        )
        segtiles = SegTiles.from_rescale(intiles, scale, fill)
        intiles.segtiles = segtiles
        segtiles = intiles.segtiles

        assert intiles.padded.segtile.index.isin(segtiles.padded.index).all()
        assert segtiles.padded.index.isin(intiles.padded.segtile.index).all()

        assert segtiles.tile.scale == scale
        assert len(segtiles) <= len(intiles)
        assert len(self) <= len(intiles)

        area = 4 ** (self.tile.scale - scale)
        assert len(intiles) == len(segtiles) * area
        assert_perfect_overlap(segtiles, intiles)

        assert segtiles.index.difference(intiles.segtile.index).empty

        return intiles

    def set_vectorization(
            self,
            *,
            dimension: int = None,
            length: int = None,
            mosaic: int = None,
            scale: int = None,
            fill: bool = True,
    ) -> Self:
        """
        dimension:
            dimension in pixels of each vectorization tile,
            including padding
        length:
            length in segmentation tiles of each vectorization tile,
            including padding
        scale:

        """
        # todo: if all are None, determine dimension using RAM
        segtiles = self.segtiles
        if dimension:
            dimension -= 2 * segtiles.tile.dimension
        elif length:
            assert length >= 3
            length -= 2
            # length *= segtiles.tile.length
        elif mosaic:
            raise NotImplementedError
            mosaic **= 1 / 2
            mosaic -= 2
            mosaic **= 2
            mosaic = int(mosaic)

        scale = self.segtiles._to_scale(dimension, length, mosaic, scale)

        msg = 'Padding InTiles to align with VecTiles'
        logger.debug(msg)
        intiles = (
            self
            .to_scale(scale, fill=fill)
            .to_scale(self.tile.scale, fill=fill)
        )

        assert intiles.tile.scale == self.intiles.tile.scale
        msg = 'Padding SegTiles to align with VecTiles'
        logger.debug(msg)
        segtiles = (
            self.segtiles
            .to_scale(scale, fill=fill)
            .to_scale(self.segtiles.tile.scale, fill=fill)
        )

        segtiles.segtiles.xtile
        self.segtile.xtile
        self.segtiles.ytile

        intiles.segtiles = segtiles

        assert intiles.padded.segtile.index.isin(segtiles.padded.index).all()
        assert segtiles.padded.index.isin(intiles.padded.segtile.index).all()

        segtiles.padded
        intiles.padded
        intiles.segtile.xtile
        assert segtiles.tile.scale == self.segtiles.tile.scale
        vectiles = VecTiles.from_rescale(intiles, scale, fill=fill)

        intiles.vectiles = vectiles
        segtiles = intiles.segtiles
        vectiles = intiles.vectiles

        assert len(self) <= len(intiles)
        assert len(vectiles) <= len(segtiles) <= len(intiles)
        area = 4 ** (self.tile.scale - scale)
        assert len(intiles) == len(vectiles) * area

        return intiles

    # @cached.
    # def network(self) -> gpd.GeoDataFrame:
    #     ...
    #
    # @property
    # def polygons(self) -> gpd.GeoDataFrame:
    #     ...

    # @property
    # def skip(self) -> pd.Series:
    #     key = 'skip'
    #     if key in self:
    #         return self[key]
    #     self[key] = self.indir.skip()
    #     return self[key]

    @Tile
    def tile(self):
        ...

    @File
    def file(self):
        ...

    @property
    def intiles(self) -> InTiles:
        return self

    def preview(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = 'red',
    ) -> PIL.Image.Image:

        files: pd.Series = self.file.infile
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

    @delayed.Padded
    def padded(self) -> Padded:
        ...

    def _make_session(
            self,
            *,
            pool: int,
            retry: Retry,
    ) -> requests.Session:
        s = requests.Session()
        s.mount(
            "https://",
            HTTPAdapter(
                pool_connections=pool,
                pool_maxsize=pool,
                max_retries=retry,
            ),
        )
        s.headers.update({"User-Agent": "tiles"})
        s.verify = certifi.where()
        return s

    def _init_net_worker(self) -> None:
        retry_head = Retry(
            total=3,
            backoff_factor=0.2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD"],
        )
        retry_get = Retry(
            total=3,
            backoff_factor=0.4,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        tls.s_head = self._make_session(pool=2, retry=retry_head)  # HEAD burst
        tls.s_get = self._make_session(pool=8, retry=retry_get)  # one stream at a time

    def _make_session(
            self,
            *,
            pool: int,
            retry: Retry,
    ) -> requests.Session:
        """
        One reusable Session with a bounded socket-pool.
        `pool_block=True` ⇒ if all connections are busy the thread *waits*
        instead of opening a new source port (prevents TIME_WAIT storms).
        """
        s = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool,
            pool_maxsize=pool,
            pool_block=True,  # ← back-pressure, not more sockets
            max_retries=retry,
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "tiles"})
        s.verify = certifi.where()
        return s

    def _init_net_worker(self) -> None:
        """
        Run once per thread → attach one Session to thread-local storage.
        Only a GET Session is needed now that the HEAD phase is gone.
        """
        retry_get = Retry(
            total=3,
            backoff_factor=0.4,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        tls.session = self._make_session(pool=8, retry=retry_get)

    @staticmethod
    def _head(url: str) -> int:  # signature unchanged
        try:
            return (
                tls.s_head
                .head(url, timeout=(3, 10))
                .status_code
            )
        except requests.RequestException:
            return -1

    @staticmethod
    def _stream_get(url: str) -> requests.Response:
        return tls.s_get.get(url, stream=True, timeout=(3, 30))

    @staticmethod
    def _atomic_write(
            path: Path,
            resp: requests.Response,
            /,
            chunk: int = 8192,
    ) -> Path | None:  # returns failed path or None

        with tempfile.NamedTemporaryFile(
                dir=path.parent,
                suffix=".tmp",
                delete=False,
        ) as tmp:
            for block in resp.iter_content(chunk):
                tmp.write(block)
            tmp_path = Path(tmp.name)

        hdr_len = int(resp.headers.get("Content-Length", -1))
        if hdr_len > 0 and tmp_path.stat().st_size != hdr_len:
            tmp_path.unlink(missing_ok=True)
            return path  # size mismatch ➜ fail

        try:
            iio.imread(tmp_path)  # quick validity check
        except ValueError:
            tmp_path.unlink(missing_ok=True)
            return path

        shutil.move(tmp_path, path)
        return None
