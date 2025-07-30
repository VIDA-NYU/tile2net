
from __future__ import annotations

import threading

from ..grid.grid import Grid

tls = threading.local()

import os
import os.path
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import delayed
import certifi
import imageio.v3 as iio
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from tile2net.grid.cfg.logger import logger
from tile2net.grid.dir.indir import Indir
from tile2net.grid.dir.outdir import Outdir

import pandas as pd

from .. import frame

from typing import *

from functools import *

from ...grid.util import recursion_block
from .polygons import Polygons
from .vectile import VecTile
from .lines import  Lines
from ..seggrid.seggrid import SegGrid
from ..grid import file
from .source import Source


class File(
    file.File
):
    grid: InGrid

    @frame.column
    def infile(self) -> pd.Series:
        grid = self.grid
        key = 'file.infile'
        if key in grid:
            return grid[key]
        files = grid.indir.files(grid)
        grid[key] = files
        if (
                not grid.download
                and not files.map(os.path.exists).all()
        ):
            grid.download()
        return grid[key]



class InGrid(Grid):
    @cached_property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        try:
            sample = next(
                p
                for p in self.grid.file.infile
                if Path(p).is_file()
            )
        except StopIteration:
            raise FileNotFoundError('No image files found to infer dimension.')
        return iio.imread(sample).shape[1]  # width

    @Indir
    def indir(self):
        ingrid = self.ingrid.outdir.ingrid
        extension = self.source.extension
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
        See `Tiles.with_source()` to actually set a source.
        """
        # This code block is just semantic sugar and does not run.
        # These methods are how to set the source:
        ingrid = self.set_source(...)  # automatically sets the source
        ingrid = self.set_source('nyc')


    @Outdir
    def outdir(self):
        ...

    @Lines
    def lines(self):
        ...

    @Polygons
    def polygons(self):
        ...

    @SegGrid
    def seggrid(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the grid have been stitched:
        _ = (
            # xtile of the larger mosaic
            self.seggrid.xtile,
            # ytile of the larger mosaic
            self.seggrid.ytile,
            # row of the tile within the larger mosaic
            self.seggrid.r,
            # column of the tile within the larger mosaic
            self.seggrid.c,
        )

    @VecTile
    def vectile(self):
        # This code block is just semantic sugar and does not run.
        _ = (
            # xtile of the larger mosaic
            self.seggrid.xtile,
            # ytile of the larger mosaic
            self.seggrid.ytile,
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

    @delayed.Padded
    def padded(self) -> Padded:
        ...

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
