from __future__ import annotations

import copy
import os
import os.path
import shutil
import sys
from ..util import ensure_tempdir_for_indir
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import *
from pathlib import Path
from typing import *

import certifi
import geopandas as gpd
import imageio.v3 as iio
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from tile2net.grid.cfg.logger import logger
from tile2net.grid.dir.indir import Indir
from tile2net.grid.dir.outdir import Outdir
from . import delayed
from .lines import Lines
from .polygons import Polygons
from .segtile import SegTile
from .source import Source, SourceNotFound
from .vectile import VecTile
from .. import frame
from ..cfg import cfg
from ..dir.dir import Dir, ExtensionNotFoundError, XYNotFoundError
from ..dir.tempdir import TempDir
from ..grid import file
from ..grid.grid import Grid
from ..grid.static import Static
from ..seggrid.seggrid import SegGrid
from ..vecgrid.vecgrid import VecGrid
from ...grid.util import recursion_block, assert_perfect_overlap

# thread-local store
tls = threading.local()

if False:
    from .padded import Padded


class File(
    file.File
):
    grid: InGrid

    @frame.column
    def infile(self) -> pd.Series:
        grid = self.grid
        files = grid.indir.files(grid)
        if (
                not grid.download
                and not files.map(os.path.exists).all()
        ):
            grid.download()
        return files


class InGrid(
    Grid
):

    @property
    def ingrid(self) -> InGrid:
        return self

    @File
    def file(self):
        ...

    @VecGrid
    def vecgrid(self) -> VecGrid:
        """
        After performing InGrid.set_vectorization(), InGrid.vecgrid is
        available for performing vectorization on the stitched tiles.
        """

    @SegGrid
    def seggrid(self) -> SegGrid:
        """
        After performing InGrid.set_segmentation(), InGrid.seggrid is
        available for performing segmentation on the stitched tiles.
        """

    @cached_property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        try:
            sample = next(
                p
                for p in self.file.infile
                if Path(p).is_file()
            )
        except StopIteration:
            self.download(one=True)
            try:
                sample = next(
                    p
                    for p in self.file.infile
                    if Path(p).is_file()
                )
            except StopIteration:
                raise FileNotFoundError('No image files found to infer dimension.')
        result = iio.imread(sample).shape[1]  # width
        return result

    @Lines
    def lines(self):
        ...

    @Polygons
    def polygons(self):
        ...

    @Static
    def static(self):
        ...

    @Indir
    def indir(self):
        ingrid = self.ingrid.outdir.ingrid
        format = os.path.join(
            ingrid.dir,
            'infile',
            f'z/x_y'
        )
        result = Indir.from_format(format)
        return result

    @Source
    def source(self):
        """
        Returns the Source class, which wraps a tile server.
        See `Grid.with_source()` to actually set a source.
        """
        # This code block is just semantic sugar and does not run.
        # These methods are how to set the source:
        ingrid = self.set_source(...)  # automatically sets the source
        ingrid = self.set_source('nyc')

    @Outdir
    def outdir(self):
        ...

    @TempDir
    def tempdir(self):
        format = ensure_tempdir_for_indir(
            self.indir.dir
        ).__str__()
        format = os.path.join(
            format,
            self.indir.suffix
        )
        result = TempDir.from_format(format)
        return result

    @SegTile
    def segtile(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the grid have been stitched:
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
            self.vectile.xtile,
            # ytile of the larger mosaic
            self.vectile.ytile,
        )

    @recursion_block
    def download(
            self,
            retry: bool = True,
            force: bool = False,
            max_workers: int = 64,  # ⇣ lower default ⇣ avoids port-exhaustion
            one: bool = False
    ) -> Self:

        # if self is not self.padded:
        #     return self.padded.download(
        #         retry=retry,
        #         force=force,
        #         max_workers=max_workers,
        #     )

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
            logger.info("All grid already on disk.")
            return self

        if not force:
            paths = paths[~exists]
            urls = urls[~exists]

        mapping = dict(zip(paths.array, urls.array))
        if not mapping:
            return self

        msg = (
            f'Downloading {len(mapping):,} '
            f'{self.__class__.__qualname__}.{self.file.infile.name} '
            f'from {self.source.name} to {self.indir.dir} '
        )
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
                ok = fut.result()
                if ok and one:
                    success = True
                    for f in futures:
                        f.cancel()
                    break

        if one and success:
            logger.info("Downloaded one file successfully (one=True).")
            return self

        if len(not_found) + len(failed) == len(self.file.infile):
            msg = f'All {len(not_found) + len(failed):,} grid failed to download.'
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
            logger.warning("%d grid returned 404 – linking placeholder.", len(not_found))
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
            raise FileNotFoundError(f"{len(failed):,} grid failed.")

        logger.info("All requested grid are on disk.")
        return self

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
        s.headers.update({"User-Agent": "grid"})
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

    @delayed.Padded
    def padded(self) -> Padded:
        ...

    def set_source(
            self,
            source=None,
            outdir: Union[str, Path] = None,
    ) -> Self:
        """
        Assign a source to the grid. The grid are downloaded from
        the source and saved to an input directory.

        You can pass an outdir which will be the destination the files
        are saved to.
        """
        if source is None:
            source = self.cfg.source
        if source is None:
            try:
                source = (
                    gpd.GeoSeries(
                        [self.frame.union_all()],
                        crs=self.frame.crs
                    )
                    .to_crs(4326)
                    .pipe(Source.from_inferred)
                )
            except SourceNotFound as e:
                msg = f'Unable to infer a source for {self.location=}'
                raise ValueError(msg) from e
        elif isinstance(source, Source):
            source = copy.copy(source)
        else:
            try:
                source = Source.from_inferred(source)
            except SourceNotFound as e:
                msg = (
                    f'Unable to infer a source from {source}. '
                    f'Please specify a valid source or use a '
                    f'different method to set the grid.'
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
            dir = outdir
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

        # infile = result.outdir.ingrid.infile
        # indir = infile.format.replace(infile.extension, source.extension)
        # result = result.set_indir(indir)

        return result

    def set_indir(
            self,
            indir: str = None,
            name: str = None,
    ) -> Self:
        """
        Assign an input directory to the grid. The directory must
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
        if indir is None:
            indir = self.cfg.indir
        result = self.copy()
        if name:
            result.name = name
        try:
            result.indir = indir
            # result.outdir.ingrid.infile = indir
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
        Assign an output directory to the grid.
        The grid are saved to the output directory.
        """
        result: Grid = self.copy()

        # if not outdir:
        # if not outdir:
        #     if cfg.output:
        #         outdi = cfg.output

        if not outdir:
            outdir = cfg.outdir

        if not outdir:
            raise ValueError(f'No output directory specified. ')

        dir = outdir
        try:
            result.outdir = dir
        except XYNotFoundError as e:
            _dir = f'{outdir}/z/x_y'
            msg = (
                f'XYZ format not implicated in given outdir format: '
                f'{dir}. Defaulting to {_dir}'
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
            fill: bool = None,
            batch_size: int = None,
    ) -> Self:
        from ..seggrid import SegGrid
        # todo: if all are None, determine dimension using VRAM

        if dimension or length or scale:
            # directly passed
            ...
        elif (
                cfg.segment.dimension !=
                cfg._default.segment.dimension
        ):
            # dimension in config
            dimension = cfg.segment.dimension
        elif (
                cfg.segment.length !=
                cfg._default.segment.length
        ):
            # length in config
            length = cfg.segment.length
        elif (
                cfg.segment.scale !=
                cfg._default.segment.scale
        ):
            # scale in config
            scale = cfg.segment.scale
        else:
            # use defaults
            dimension = cfg.segment.dimension
            length = cfg.segment.length
            scale = cfg.segment.scale

        scale = self._to_scale(dimension, length, mosaic, scale)

        if batch_size:
            self.cfg.model.bs_val = batch_size
        if fill is None:
            fill = self.cfg.segment.fill

        msg = 'Padding InGrid to align with SegGrid'
        logger.debug(msg)
        ingrid = (
            self
            .to_scale(scale, fill=fill)
            .to_scale(self.scale, fill=fill)
        )
        seggrid = SegGrid.from_rescale(ingrid, scale, fill)
        ingrid.seggrid = seggrid
        seggrid = ingrid.seggrid

        assert (
            ingrid.padded.segtile.index
            .isin(seggrid.padded.index)
            .all()
        )
        assert (
            seggrid.padded.index
            .isin(ingrid.padded.segtile.index)
            .all()
        )

        assert seggrid.scale == scale
        assert len(seggrid) <= len(ingrid)
        assert len(self) <= len(ingrid)

        area = 4 ** (self.scale - scale)
        assert len(ingrid) == len(seggrid) * area
        assert_perfect_overlap(seggrid, ingrid)

        assert seggrid.index.difference(ingrid.segtile.index).empty

        return ingrid

    def set_vectorization(
            self,
            *,
            dimension: int = None,
            length: int = None,
            # mosaic: int = None,
            scale: int = None,
            fill: bool = True,
    ) -> Self:
        """
        dimension:
            dimension in pixels of each vectorization tile,
            including padding
        length:
            length in segmentation grid of each vectorization tile,
            including padding
        scale:
        """

        if dimension or length or scale:
            # directly passed
            ...
        elif (
                cfg.vector.dimension !=
                cfg._default.vector.dimension
        ):
            # dimension in config
            dimension = cfg.vector.dimension
        elif (
                cfg.vector.length !=
                cfg._default.vector.length
        ):
            # length in config
            length = cfg.vector.length
        elif (
                cfg.vector.scale !=
                cfg._default.vector.scale
        ):
            # scale in config
            scale = cfg.vector.scale
        else:
            # use defaults
            dimension = cfg.vector.dimension
            length = cfg.vector.length
            scale = cfg.vector.scale

        # todo: if all are None, determine dimension using RAM
        seggrid = self.seggrid
        if dimension:
            dimension -= 2 * seggrid.dimension
        if length:
            assert length >= 3
            length -= 2
            # length *= seggrid.length
        # if mosaic:
        #     raise NotImplementedError
        #     mosaic **= 1 / 2
        #     mosaic -= 2
        #     mosaic **= 2
        #     mosaic = int(mosaic)

        scale = self.seggrid._to_scale(dimension, length, scale)

        msg = 'Padding InGrid to align with VecGrid'
        logger.debug(msg)
        ingrid = (
            self
            .to_scale(scale, fill=fill)
            .to_scale(self.scale, fill=fill)
        )

        assert ingrid.scale == self.ingrid.scale
        msg = 'Padding SegGrid to align with VecGrid'
        logger.debug(msg)
        seggrid = (
            self.seggrid
            .to_scale(scale, fill=fill)
            .to_scale(self.seggrid.scale, fill=fill)
        )

        ingrid.seggrid = seggrid

        assert ingrid.padded.segtile.index.isin(seggrid.padded.index).all()
        assert seggrid.padded.index.isin(ingrid.padded.segtile.index).all()
        assert seggrid.scale == self.seggrid.scale
        vecgrid = VecGrid.from_rescale(ingrid, scale, fill=fill)

        ingrid.vecgrid = vecgrid
        seggrid = ingrid.seggrid
        vecgrid = ingrid.vecgrid

        assert len(self) <= len(ingrid)
        assert len(vecgrid) <= len(seggrid) <= len(ingrid)
        area = 4 ** (self.scale - scale)
        assert len(ingrid) == len(vecgrid) * area

        return ingrid

    def summary(self) -> None:
        # prominent end-of-run banner with aligned columns
        # helpers
        def _p(v) -> Path | None:
            if v is None:
                return None
            p = Path(v)
            return p if str(p).strip() else None

        def _abs(p: Path) -> str:
            return str(p.resolve())

        # collect resources
        rows = [
            ('Input imagery', _p(self.outdir.ingrid.infile.dir)),
            ('Segmentation (colored)', _p(self.outdir.seggrid.colored.dir)),
            # ('Polygons by tile',    _p(self.outdir.vecgrid.polygons.dir)),
            # ('Lines by tile',       _p(self.outdir.vecgrid.lines.dir)),
            # ('HRNet checkpoint',           _p(self.static.hrnet_checkpoint)),
            # ('Model snapshot',             _p(self.static.snapshot)),
            ('Polygons', _p(self.outdir.polygons.file)),
            ('Network', _p(self.outdir.lines.file)),
        ]

        # compute column widths
        label_w = max(len(k) for k, _ in rows)
        term_w = shutil.get_terminal_size((100, 20)).columns
        sep = '=' * min(term_w, 80)

        # color if TTY
        use_color = sys.stdout.isatty()
        BOLD = '\033[1m' if use_color else ''
        DIM = '\033[2m' if use_color else ''
        GRN = '\033[32m' if use_color else ''
        CYN = '\033[36m' if use_color else ''
        YEL = '\033[33m' if use_color else ''
        RST = '\033[0m' if use_color else ''

        # header
        print(sep)
        print(f"{BOLD}Tile2Net run complete!{RST}")
        outdir = _p(self.outdir.dir if hasattr(self.outdir, 'dir') else self.outdir)
        if outdir is None and hasattr(self.outdir, 'root'):
            outdir = _p(self.outdir.root)
        if outdir is not None:
            print(f"{CYN}Output directory:{RST} {_abs(outdir)}")
        print(sep)

        # body
        for label, path in rows:
            if path is None:
                line = f"{label:<{label_w}}  {DIM}— not set —{RST}"
            else:
                exists = path.exists()
                path_str = _abs(path)
                if exists:
                    color = GRN if path.is_dir() else CYN
                    line = f"{label:<{label_w}}  {color}{path_str}{RST}"
                else:
                    line = f"{label:<{label_w}}  {YEL}{path_str}{RST} {DIM}(missing){RST}"
            print(line)
