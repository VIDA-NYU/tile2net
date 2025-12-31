from __future__ import annotations

import copy
import hashlib
import os
import os.path
import pickle
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import *
from pathlib import Path
from typing import *

import certifi
import geopandas as gpd
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import ImageColor, Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from tile2net.grid.cfg.logger import logger
from tile2net.grid.dir.indir import Indir
from tile2net.grid.dir.outdir import Outdir
from . import delayed
from .file import File
from .lines import Lines
from .polygons import Polygons
from .segtile import SegTile
from .source import Source, SourceNotFound
from .vectile import VecTile
from ..cfg import cfg
from ..cfg.cfg import Cfg
from ..dir.dir import Dir, ExtensionNotFoundError, XYNotFoundError
from ..dir.tempdir import TempDir
from ..grid.grid import Grid
from ..grid.static import Static
from ..loaders.datawrapper import DataWrapper
from ..loaders.rescale import RescaleDataSet
from ..sampler.benchmark import Benchmark
from ..seggrid.seggrid import SegGrid
from ..vecgrid.vecgrid import VecGrid
from ...grid import util
from ...grid.util import recursion_block, assert_perfect_overlap

if False:
    from .filled import Filled
    from .broadcast import Broadcast
    from . import _pickle, construct

# thread-local store
tls = threading.local()


@dataclass
class Pickle:
    path: Path

    @cached_property
    def md5(self) -> str:
        h = hashlib.md5()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        result = f'md5:{h.hexdigest()}'
        return result


def file_md5(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()




class InGrid(
    Grid
):
    """
    "Input Grid" (InGrid), comprised of "input tiles" (in-tiles).
    Each tile is an image from the source.

    Example construction:
        >>> ingrid = InGrid.from_location('Boston Common, MA')
        InGrid:
                             lonmin        latmax        lonmax        latmin
        xtile  ytile
        317280 387840 -7.911538e+06  5.214840e+06 -7.911500e+06  5.214802e+06

    Handles downloading of input tiles:
        >>> InGrid.download

    """

    @File
    def file(self):
        """
        Namespace container for files aligned with the tiles of a Grid.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.file.infile
            xtile   ytile
            317280  387840    /home/<user>/tile2net/ma/ingrid/infile/20/31...
        """

    @VecGrid
    def vecgrid(self) -> VecGrid:
        """
        Wraps vectorization operations with a grid data structure

        After performing InGrid.set_vectorization(), InGrid.vecgrid is
        available for performing vectorization on the stitched tiles.

        Example:
            >>> InGrid.vecgrid
            VecGrid:
                       lonmin        latmax        lonmax        latmin  \
            xtile ytile
            9915  12120 -7.911538e+06  5.214840e+06 -7.910315e+06  5.213617e+06
                                                                  geometry
            xtile ytile
            9915  12120  POLYGON ((-7910315.183 5214839.818, -7910315.1...

        Handles lazy-loading of InGrid.VecGrid:
            >>> VecGrid._get

        Handles vectorization process from stitched seg-tiles:
            >>> VecGrid.vectorize
        """

    @SegGrid
    def seggrid(self) -> SegGrid:
        """
        Wraps segmentation prediction operations with a grid data structure

        After performing InGrid.set_segmentation(), InGrid.seggrid is
        available for performing segmentation on the stitched tiles.

        Example:
            >>> InGrid.seggrid
            SegGrid:
                           lonmin        latmax        lonmax        latmin  \
            xtile ytile
            79320 96960 -7.911538e+06  5.214840e+06 -7.911385e+06  5.214687e+06
                                                                  geometry
            xtile ytile
            79320 96960  POLYGON ((-7911385.302 5214839.818, -7911385.3...
            [64 rows x 5 columns]

        Handles lazy-loading of InGrid.SegGrid:
        >>> SegGrid._get

        Handles prediction of seg-tiles from stitched input tiles:
        >>> SegGrid._predict
        """

    @cached_property
    def dimension(self) -> int:
        """
        Tile dimension in pixels; inferred from input files.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.dimension
            256
        """
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

    @cached_property
    def name(self) -> str:
        """
        Grid name; inferred from config, location, or input directory.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.name
            'Boston Common, MA'
        """
        name = (
                self.cfg.name
                or self.cfg.indir.name
                or self.location
                or self.indir.dir.rsplit(os.sep, 1)[-1]
        )
        return name

    @property
    def shape(self):
        """
        Tile shape as (height, width) in pixels.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.shape
            (256, 256)
        """
        return self.dimension, self.dimension

    @Lines
    def lines(self):
        """
        Lines from all features (e.g. sidewalks, crosswalks) from all tiles dissolved into
        continuous line geometries.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.lines
            Lines:
                                                    geometry
            feature
            crosswalk  LINESTRING (-7910926 5213692.6, -7910925.6 521...
        """

    @Polygons
    def polygons(self):
        """
        Polygons from all features (e.g. sidewalks, crosswalks) from all tiles dissolved into
        continuous polygon geometries.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.polygons
            Polygons:
                                                    geometry
            feature
            crosswalk  POLYGON ((-7911335.6 5213618.8, -7911339.8 521...
        """

    @Static
    def static(self):
        """
        Namespace for static assets (e.g., placeholder images, weights).

        Example:
            >>> ingrid: InGrid
            >>> ingrid.static.black
            >>> ingrid.static.hrnet_checkpoint
            >>> ingrid.static.snapshot
        """

    @Indir
    def indir(self):
        """
        Input Directory: location at which input imagery are stored and read from.
        If using a remote source, images will be downloaded to this directory.


        # Sets the input directory:
        >>> InGrid.set_indir
        """
        raise ValueError('No input directory specified. ')
        ingrid = self.ingrid.outdir.ingrid
        format = os.path.join(
            ingrid.dir,
            'infile',
            f'z/x_y'
        )
        result = Indir.from_format(format)
        return result

    @Outdir
    def outdir(self):
        """
        Output in which the results, such as annotated images and geometry, will be stored:
        
        >>> ingrid: InGrid
        >>> Outdir(
        >>>     format='/home/<user>/tile2net/{z}/{x}_{y}',
        >>>     dir='/home/<user>/tile2net',
        >>>     original='/home/<user>/tile2net/z/x_y',
        >>>     suffix='z/x_y'
        >>> )

        Setting the output directory:
        >>> ingrid: InGrid
        >>> ingrid = ingrid.set_outdir('/path/to/output')
        """

    @Source
    def source(self):
        """
        Returns the Source class, which wraps a tile server.
        See `Grid.with_source()` to actually set a source.

        Automatically sets the source:
        >>> ingrid: InGrid
        >>> ingrid = ingrid.set_source(...)
        """

    @delayed.Pickle
    def pickle(self) -> _pickle.Pickle:
        """
        Module which offers pre-constructed `InGrid` instances. This can save time during testing.
        """

    @delayed.Construct
    def construct(self) -> construct.Construct:
        """
        Module which offers pre-constructed `InGrid` instances.
        """


    @TempDir
    def tempdir(self):
        """
        Temporary directory for intermediate processing files.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.tempdir
            Tempdir(
                dir='/tmp/tile2net/ma/ingrid/infile'
                original='/tmp/tile2net/ma/ingrid/infile/z/x_y',
            )
        """
        format = os.path.join(
            tempfile.gettempdir(),
            'tile2net',
            self.indir.suffix
        )
        result = TempDir.from_format(format)
        return result

    @SegTile
    def segtile(self):
        """
        Namespace for seg-tile properties aligned with in-tiles.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.xtile
            xtile   ytile
            317280  387840    79320
        """

    @VecTile
    def vectile(self):
        """
        Namespace for vec-tile properties aligned with in-tiles.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vectile.xtile
            xtile   ytile
            317280  387840    9915
        """

    @recursion_block
    def download(
            self,
            retry: bool = True,
            force: bool = False,
            max_workers: int = 64,  # ⇣ lower default ⇣ avoids port-exhaustion
            one: bool = False
    ) -> Self:
        """
        Download tiles from the configured source to the input directory.

        Args:
            retry:
                Retry failed downloads once
            force:
                Re-download existing files
            max_workers:
                Maximum concurrent download threads
            one:
                Download only one file for testing

        Returns:
            Self with downloaded files available at InGrid.file.infile
        """

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
            msg = (
                f'All {len(paths):,} '
                f'{self.__class__.__qualname__}.{self.file.infile.name} '
                f'on disk at {self.indir.dir}. '
            )
            logger.info(msg)
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
            f'from {self.source.name} to \n\t{self.indir.dir} '
        )
        logger.info(msg)

        pool_size = min(max_workers, len(mapping))
        not_found: list[Path] = []
        failed: list[Path] = []
        success = False

        def _fetch_write(
                path: Path,
                url: str,
        ) -> None:
            # use Response as a context manager so the socket/FD is always released
            try:
                with tls.session.get(url, stream=True, timeout=(3, 30)) as resp:
                    status = resp.status_code
                    if status == 404:
                        # not found: nothing to write
                        not_found.append(path)
                        return
                    if status == 403:
                        # forbidden: treat like not found → link placeholder later
                        not_found.append(path)
                        return
                    if status != 200:
                        # other error: mark as failed
                        failed.append(path)
                        return

                    # atomic write into a sibling temp file
                    fd, tmp_path = tempfile.mkstemp(
                        dir=path.parent,
                        suffix=".part",
                    )
                    os.close(fd)
                    tmp = Path(tmp_path)

                    try:
                        # stream body to disk
                        with tmp.open("wb") as fh:
                            for chunk in resp.iter_content(1 << 15):
                                if not chunk:
                                    continue
                                fh.write(chunk)

                        # quick sanity check
                        try:
                            iio.imread(tmp)
                        except ValueError:
                            tmp.unlink(missing_ok=True)
                            failed.append(path)
                            return

                        # promote temp into place
                        shutil.move(tmp, path)

                    except Exception:
                        # any unexpected error during write → remove temp
                        tmp.unlink(missing_ok=True)
                        failed.append(path)
                        return

            except Exception:
                # network/requests‑level failure before we could write
                failed.append(path)
                return

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
                    unit=f" {self.__class__.__name__}.{paths.name}",
                    mininterval=10,
            ):
                fut.result()

                if one:
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
            black = self.static.black
            logger.warning("%d tile(s) returned 404/403 – linking placeholder.", len(not_found))
            for p in not_found:
                try:
                    os.symlink(black, p)
                except (OSError, NotImplementedError):
                    shutil.copy2(black, p)

        if failed:
            if retry:
                logger.error("%d tile(s) failed; retrying once.", len(failed))
                return self.download(
                    retry=False,
                    force=False,
                    max_workers=max_workers,
                )
            raise FileNotFoundError(f"{len(failed):,} files failed.")

        msg = (
            f"All requested {self.__class__.__name__}.{paths.name} "
            f"on disk at \n\t{self.indir.dir} "
        )
        logger.info(msg)
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

    @delayed.Filled
    def filled(self) -> Filled:
        """
        "Fills" the InGrid with extra tiles implicated by the VecGrid to avoid missing tiles.
        For example, if the InGrid is 1x1, but the VecGrid tiles are supposed to be 2 in-tiles wide,
        this will fill in the missing tiles so there are 2x2 total tiles in the InGrid.
        """

    @delayed.Broadcast
    def broadcast(self) -> Broadcast:
        """
        While a seg-tile is comprised of in-tiles, an in-tile may belong to multiple seg-tiles due to
        overlaps. The base InGrid dataframe has one row per unique in-tile. The Broadcast extension
        overcomes this limitation by possibly having multiple rows per in-tile.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.broadcast

                        frame.sort_index()
            Out[17]:
                           segtile.xtile  segtile.ytile  segtile.r  segtile.c
            xtile  ytile
            317275 387839          79319          96959          4          0
                   387839          79319          96960          0          0
        """

    def set_source(
            self,
            source=None,
            outdir: Union[str, Path] = None,
    ) -> Self:
        """
        Assign a tile source for downloading imagery.

        Args:
            source: Source name, abbreviation, or Source instance. If None, infers from location.
            outdir: Optional output directory path

        Returns:
            InGrid with source and directories configured

        Example:
            >>> ingrid: InGrid = InGrid.from_location('Boston Common, MA').set_source('ma')
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

        # if not outdir:
        #     outdir = (
        #         Path(f'./{source.name}')
        #         .expanduser()
        #         .resolve()
        #         .__str__()
        #     )
        if not outdir:
            outdir = (
                Path(cfg.outdir)
                .expanduser()
                .resolve()
                .__str__()
            )

        try:
            dir = Dir.from_format(outdir)
        except ExtensionNotFoundError:
            dir = outdir
            try:
                dir = Dir.from_format(dir)
            except XYNotFoundError as e:
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

        outdir = dir.original

        result = result.set_outdir(outdir)

        ingrid = result.outdir.ingrid
        format = os.path.join(
            ingrid.dir,
            'infile',
            f'z/x_y'
        )
        result = result.set_indir(format)

        return result

    def set_indir(
            self,
            indir: str = None,
            name: str = None,
    ) -> Self:
        """
        Assign an input directory where tiles are stored.

        The directory path must include x and y tile coordinate placeholders.

        Args:
            indir: Directory path with x/y format (e.g., 'input/dir/z/x_y.png')
            name: Optional grid name

        Returns:
            InGrid with input directory configured

        Example:
            >>> ingrid: InGrid = ingrid.set_indir('/path/to/tiles/20/x_y.png')
        """
        if indir is None:
            indir = self.cfg.indir.path
        result = self.copy()
        if name:
            result.name = name
        try:
            result.indir = indir
            # result.outdir.ingrid.infile = indir
            indir: Indir = result.indir
            msg = f'Setting input directory to \n\t{indir.original} '

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
        Assign an output directory for processed results.

        Args:
            outdir: Output directory path

        Returns:
            InGrid with output directory configured

        Example:
        >>> ingrid: InGrid
        >>> ingrid: InGrid = ingrid.set_outdir('/path/to/output')
        """
        result: Grid = self.copy()

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
            pad=None,
    ) -> Self:
        """
        Configure segmentation grid dimensions and create InGrid.seggrid.

        Args:
            dimension: Pixel dimension of each seg-tile
            length: Number of in-tiles per seg-tile dimension
            scale: Zoom scale for seg-tiles
            fill: Whether to fill missing tiles
            batch_size: Batch size for model inference
            pad: Padding pixels for seg-tiles

        Returns:
            InGrid with InGrid.seggrid configured

        Example:
            >>> ingrid: InGrid = ingrid.set_segmentation(dimension=1024, pad=64)
        """
        from ..seggrid import SegGrid
        # todo: if all are None, determine dimension using VRAM

        if dimension or length or scale:
            # directly passed
            ...
        elif (
                cfg.segmentation.dimension !=
                cfg._default.segmentation.dimension
        ):
            # dimension in config
            dimension = cfg.segmentation.dimension
        elif (
                cfg.segmentation.length !=
                cfg._default.segmentation.length
        ):
            # length in config
            length = cfg.segmentation.length
        elif (
                cfg.segmentation.scale !=
                cfg._default.segmentation.scale
        ):
            # scale in config
            scale = cfg.segmentation.scale
        else:
            # use defaults
            dimension = cfg.segmentation.dimension
            length = cfg.segmentation.length
            scale = cfg.segmentation.scale

        scale = self._to_scale(dimension, length, mosaic, scale)

        if batch_size:
            self.cfg.model.bs_val = batch_size
        if fill is None:
            fill = self.cfg.segmentation.fill

        msg = 'Filling InGrid to align with SegGrid'
        logger.debug(msg)
        ingrid = (
            self
            .to_scale(scale, fill=fill)
            .to_scale(self.scale, fill=fill)
        )
        seggrid = SegGrid.from_rescale(ingrid, scale, fill)
        ingrid.seggrid = seggrid
        seggrid = ingrid.seggrid
        if pad is not None:
            seggrid.pad = pad

        assert (
            ingrid.filled.segtile.index
            .isin(seggrid.filled.index)
            .all()
        )
        assert (
            seggrid.filled.index
            .isin(ingrid.filled.segtile.index)
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
            pad: int = None,
    ) -> Self:
        """
        Configure vectorization grid dimensions and create InGrid.vecgrid.

        Args:
            dimension: Pixel dimension of each vec-tile including padding
            length: Number of seg-tiles per vec-tile dimension
            scale: Zoom scale for vec-tiles
            fill: Whether to fill missing tiles
            pad: Padding pixels for vec-tiles

        Returns:
            InGrid with InGrid.vecgrid configured

        Example:
            >>> ingrid: InGrid = ingrid.set_vectorization(length=5, pad=128)
        """

        if dimension or length or scale:
            # directly passed
            ...
        elif (
                cfg.vectorization.dimension !=
                cfg._default.vectorization.dimension
        ):
            # dimension in config
            dimension = cfg.vectorization.dimension
        elif (
                cfg.vectorization.length !=
                cfg._default.vectorization.length
        ):
            # length in config
            length = cfg.vectorization.length
        elif (
                cfg.vectorization.scale !=
                cfg._default.vectorization.scale
        ):
            # scale in config
            scale = cfg.vectorization.scale
        else:
            # use defaults
            dimension = cfg.vectorization.dimension
            length = cfg.vectorization.length
            scale = cfg.vectorization.scale

        # todo: if all are None, determine dimension using RAM
        seggrid = self.seggrid
        # if dimension:
        #     dimension -= 2 * seggrid.dimension
        # if length:
        #     assert length >= 3
        #     length -= 2
        # length *= seggrid.length
        # if mosaic:
        #     raise NotImplementedError
        #     mosaic **= 1 / 2
        #     mosaic -= 2
        #     mosaic **= 2
        #     mosaic = int(mosaic)

        scale = self.seggrid._to_scale(dimension, length, scale)

        msg = 'Filling InGrid to align with VecGrid'
        logger.debug(msg)
        ingrid = (
            self
            .to_scale(scale, fill=fill)
            .to_scale(self.scale, fill=fill)
        )

        assert ingrid.scale == self.ingrid.scale
        msg = 'Filling SegGrid to align with VecGrid'
        logger.debug(msg)
        seggrid = (
            self.seggrid
            .to_scale(scale, fill=fill)
            .to_scale(self.seggrid.scale, fill=fill)
        )

        ingrid.seggrid = seggrid

        assert ingrid.filled.segtile.index.isin(seggrid.filled.index).all()
        assert seggrid.filled.index.isin(ingrid.filled.segtile.index).all()
        assert seggrid.scale == self.seggrid.scale
        vecgrid = VecGrid.from_rescale(ingrid, scale, fill=fill)

        if pad is not None:
            vecgrid.pad = pad

        ingrid.vecgrid = vecgrid
        seggrid = ingrid.seggrid
        vecgrid = ingrid.vecgrid

        assert len(self) <= len(ingrid)
        assert len(vecgrid) <= len(seggrid) <= len(ingrid)
        area = 4 ** (self.scale - scale)
        assert len(ingrid) == len(vecgrid) * area

        return ingrid

    def summary(self) -> None:
        """
        Prints a summary of the grid's contents and performance.
        """

        # helpers
        def _p(v) -> Path | None:
            if v is None:
                return None
            p = Path(v)
            return p if str(p).strip() else None

        def _abs(p: Path) -> str:
            s = str(p.resolve())
            try:
                home = str(Path.home())
                if s == home or s.startswith(home + os.sep):
                    s = '~' + s[len(home):]
            except Exception:
                pass
            return s

        # collect resources
        rows = []

        outdir = _p(getattr(self.outdir, 'dir', None) or getattr(self.outdir, 'root', None))
        tempdir = _p(getattr(self.tempdir, 'dir', None) or getattr(self.tempdir, 'root', None))

        if outdir:
            rows.append(('Output directory', outdir))

        rows.append(('Input imagery', _p(self.outdir.ingrid.infile.dir)))
        if self.cfg.segmentation.colored:
            rows.append(('Segmentation (colored)', _p(self.outdir.seggrid.colored.dir)))
        if self.cfg.polygon.concat:
            rows.append(('Polygons', _p(self.outdir.polygons.parquet)))
        if self.cfg.line.concat:
            rows.append(('Network', _p(self.outdir.lines.parquet)))
        if self.cfg.polygon.preview:
            rows.append(('Polygon preview', _p(self.outdir.polygons.preview)))
        if self.cfg.line.preview:
            rows.append(('Network preview', _p(self.outdir.lines.preview)))
        if self.cfg.segmentation.to_pkl:
            rows.append(('seg-tiles (zoom and scale in attrs)', _p(self.outdir.seggrid.pickle)))

        # compute formatting
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
        print(sep)

        # performance summaries
        def _fmt_pct(v: float) -> str:
            return f"{v:.1f}%"

        def _fmt_duration(
                v: float
        ) -> str:
            # choose human-friendly units: d, h, m, s, ms
            if v is None:
                return "—"
            secs = float(v)
            if secs < 0:
                secs = 0.0
            ms = secs * 1000.0
            if secs < 1e-3:
                return "0s"
            if secs < 1.0:
                return f"{ms:.0f}ms"
            days = int(secs // 86400)
            secs -= days * 86400
            hours = int(secs // 3600)
            secs -= hours * 3600
            minutes = int(secs // 60)
            secs -= minutes * 60
            parts = []
            if days:
                parts.append(f"{days}d")
            if hours and len(parts) < 2:
                parts.append(f"{hours}h")
            if minutes and len(parts) < 2:
                parts.append(f"{minutes}m")
            if len(parts) < 2:
                if secs >= 10:
                    parts.append(f"{int(secs)}s")
                else:
                    parts.append(f"{secs:.1f}s")
            return " ".join(parts)

        def _print_line(label: str, avg: float | None, maxv: float | None) -> None:
            parts = []
            if avg is not None:
                parts.append(f"{DIM}avg{RST} {CYN}{_fmt_pct(avg)}{RST}")
            if maxv is not None:
                parts.append(f"{DIM}max{RST} {GRN}{_fmt_pct(maxv)}{RST}")
            if parts:
                print(f"{label}: " + " | ".join(parts))

        # segmentation summary
        try:
            seg_s = self.seggrid.benchmark.samples
            seg_vals = {
                'elapsed': seg_s.time_elapsed,
                'gpu_avg': seg_s.avg_gpu,
                'gpu_max': seg_s.max_gpu,
                'vram_avg': seg_s.avg_vram,
                'vram_max': seg_s.max_vram,
                'ram_avg': seg_s.avg_ram,
                'ram_max': seg_s.max_ram,
                'cpu_avg': seg_s.avg_cpu,
                'cpu_max': seg_s.max_cpu,
            }
            if any(v is not None for v in seg_vals.values()):
                print(sep)
                print(f"{BOLD}Segmentation benchmark{RST}")
                print(sep)
                if seg_vals['elapsed'] is not None:
                    print(f"Time Elapsed: {CYN}{_fmt_duration(seg_vals['elapsed'])}{RST}")
                _print_line("GPU Compute", seg_vals['gpu_avg'], seg_vals['gpu_max'])
                _print_line("VRAM Usage", seg_vals['vram_avg'], seg_vals['vram_max'])
                _print_line("RAM Usage", seg_vals['ram_avg'], seg_vals['ram_max'])
        except Exception:
            pass

        # vectorization summary
        try:
            vec_s = self.vecgrid.benchmark.samples
            vec_vals = {
                'elapsed': vec_s.time_elapsed,
                'ram_avg': vec_s.avg_ram,
                'ram_max': vec_s.max_ram,
                'cpu_avg': vec_s.avg_cpu,
                'cpu_max': vec_s.max_cpu,
            }
            if any(v is not None for v in vec_vals.values()):
                print(sep)
                print(f"{BOLD}Vectorization benchmark{RST}")
                print(sep)
                if vec_vals['elapsed'] is not None:
                    print(f"Time Elapsed: {CYN}{_fmt_duration(vec_vals['elapsed'])}{RST}")
                _print_line("RAM Usage", vec_vals['ram_avg'], vec_vals['ram_max'])
                _print_line("CPU Usage", vec_vals['cpu_avg'], vec_vals['cpu_max'])
        except Exception:
            pass

        # polygon concatenation time
        try:
            poly_s = self.polygon_benchmark.samples  # preferred if available
            if getattr(poly_s, 'time_elapsed', None) is not None:
                print(sep)
                print(f"{BOLD}Polygon concatenation{RST}")
                print(sep)
                print(f"Time Elapsed: {CYN}{_fmt_duration(poly_s.time_elapsed)}{RST}")
        except Exception:
            pass

        # network concatenation time
        try:
            line_s = self.line_benchmark.samples  # preferred if available
            if getattr(line_s, 'time_elapsed', None) is not None:
                print(sep)
                print(f"{BOLD}Network concatenation{RST}")
                print(sep)
                print(f"Time Elapsed: {CYN}{_fmt_duration(line_s.time_elapsed)}{RST}")
        except Exception:
            pass

        # body (two-line format + blank line between rows)
        for label, path in rows:
            if path is None:
                print(f"{label:<{label_w}}")
                print(f"{DIM}— not set —{RST}")
            else:
                path_str = _abs(path)
                if path.exists():
                    color = GRN if path.is_dir() else CYN
                    print(f"{label:<{label_w}}")
                    print(f"{color}{path_str}{RST}")
                else:
                    print(f"{label:<{label_w}}")
                    print(f"{YEL}{path_str}{RST} {DIM}(missing){RST}")
            print()  # <-- blank line between rows

    def cleanup(
            self,
            polygons: bool = True,
            lines: bool = True,
            infile: bool = True,
            grayscale: bool = True
    ):
        """ignore for now."""
        raise NotImplementedError
        self.outdir.cleanup()
        self.tempdir.cleanup()

        if polygons:
            msg = (
                f'Cleaning up the polygons for each tile from '
                f'\n\t{self.ingrid.outdir.vecgrid.polygons.dir}'
            )
            logger.info(msg)
            util.cleanup(self.vecgrid.file.polygons)

        if lines:
            msg = (
                f'Cleaning up the lines for each tile from '
                f'\n\t'
                f'{self.ingrid.outdir.vecgrid.lines.dir}'
            )
            logger.info(msg)
            util.cleanup(self.vecgrid.file.lines)

        if infile:
            msg = (
                f'Cleaning up previously downloaded imagery '
                f'from {self.ingrid.indir.dir} and '
                f'{self.ingrid.outdir.seggrid.infile.dir}'
            )
            logger.info(msg)
            util.cleanup(self.ingrid.file.infile)
            util.cleanup(self.seggrid.file.infile)

        if grayscale:
            msg = (
                f'Cleaning up segmentation masks '
                f'from \n\t{self.outdir.seggrid.grayscale.dir} and '
                f'\n\t{self.outdir.vecgrid.grayscale.dir}'
            )
            logger.info(msg)
            util.cleanup(self.seggrid.file.grayscale)
            util.cleanup(self.vecgrid.file.grayscale)

    @cached_property
    def disk_usage(self) -> int:
        """
        Total disk space used by all segmentation files in bytes.
        """
        result = self.broadcast.file.disk_usage.sum()
        return result

    @cached_property
    def time_usage(self) -> float:
        """
        Time spent on segmentation operations in seconds.
        """
        return 0.

    def preview(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = 'red',
            show: bool = True,
            files: Optional[pd.Series] = None,
    ) -> Image.Image:
        """
        Generate a mosaic preview of all tiles in the grid.

        Args:
            maxdim: Maximum dimension for the output image
            divider: Color name for tile divider lines
            show: Display the preview automatically
            files: Optional custom file paths to preview

        Returns:
            PIL Image containing the tile mosaic
        """
        # todo: divider isn't showing up

        # grid geometry
        dim = self.dimension
        R: pd.Series = self.r
        C: pd.Series = self.c

        # total rows/cols
        n_rows = int(R.max()) + 1
        n_cols = int(C.max()) + 1

        # compute full-res canvas size assuming 1px divider lines between tiles
        # (the dataset you're using below is already aware of 'divider' as background)
        w = n_cols * dim + n_cols
        h = n_rows * dim + n_rows
        m = max(w, h)

        # downscale factor to respect maxdim
        if m <= maxdim:
            scale = 1.0
        else:
            scale = maxdim / m

        # build a wrapper that knows how to place tiles with a divider-colored background
        if files is None:
            files = self.file.infile

        wrapper = DataWrapper.from_tiles(
            infile=self.file.infile,
            index=self.index,
            row=self.r,
            col=self.c,
            background=divider,
            force=True,
        )

        # dataset/loader provide batches with x0, y0, arr already scaled
        dataset = RescaleDataSet(wrapper, scale=scale, dim=dim)
        loader = dataset.loader

        # base color for the mosaic background
        base_rgb = ImageColor.getrgb(divider or 'black')

        # allocate mosaic as RGB uint8 and fill with base color
        # shape uses scaled dimensions derived implicitly from dataset outputs;
        # we allocate using the scaled canvas size to avoid incremental growth.
        # Scale the nominal w,h computed above.

        h = max(dataset.y1) + 1
        w = max(dataset.x1) + 1
        mosaic_np = np.empty((h, w, 3), dtype=np.uint8)
        mosaic_np[...] = base_rgb

        # paste tiles using numpy copy semantics
        for batch in loader:
            x0 = batch.x0
            y0 = batch.y0
            arr = batch.arr
            x1 = batch.x1
            y1 = batch.y1

            # ensure uint8 RGB
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)

            if arr.ndim == 2:
                # grayscale → RGB
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                # drop alpha
                arr = arr[:, :, :3]
            elif arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f'Unexpected tile shape: {arr.shape!r}')
            h_tile, w_tile = arr.shape[0], arr.shape[1]

            # bounds check (defensive; no-op if in-bounds)
            if (
                    y0 < 0
                    or x0 < 0
                    or y0 + h_tile > mosaic_np.shape[0]
                    or x0 + w_tile > mosaic_np.shape[1]
            ):
                raise ValueError(
                    f'Tile at ({x0},{y0}) with size ({w_tile},{h_tile}) '
                    f'exceeds mosaic bounds ({mosaic_np.shape[1]},{mosaic_np.shape[0]}).'
                )

            # fast in-place write
            np.copyto(mosaic_np[y0:y1, x0:x1, :], arr)

        # convert to PIL for downstream consumers
        mosaic_im = Image.fromarray(mosaic_np, mode='RGB')

        # optional display (PyCharm SciView / matplotlib)
        if show:
            try:
                # keep DPI moderate to avoid giant windows
                dpi = 96
                fig_w_in = max(1.0, mosaic_im.width / dpi)
                fig_h_in = max(1.0, mosaic_im.height / dpi)

                plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
                plt.imshow(mosaic_im)
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.show()

            except Exception:
                # fallback to OS image viewer
                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                try:
                    mosaic_im.save(tmp.name)
                finally:
                    tmp.close()

                try:
                    with Image.open(tmp.name) as im:
                        im.show()
                finally:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass

        return mosaic_im

    @cached_property
    def polygon_benchmark(self) -> Benchmark:
        result = Benchmark(include_gpu=True)
        return result

    @cached_property
    def line_benchmark(self) -> Benchmark:
        result = Benchmark(include_gpu=True)
        return result

    @classmethod
    def from_cfg(
            cls,
            cfg: Cfg = None
    ) -> Self:
        """
        Construct InGrid from a configuration object or JSON file.

        Args:
            cfg: Configuration object, path to JSON config, or None for CLI args

        Returns:
            Fully configured InGrid instance

        Example:
            >>> ingrid: InGrid = InGrid.from_cfg('config.json')
        """
        if isinstance(cfg, (str, Path)):
            cfg = Cfg.from_json(cfg)
        if cfg is None:
            cfg = Cfg.from_parser()
        with cfg:
            ingrid = InGrid.from_location(
                location=cfg.location,
                zoom=cfg.zoom
            )
            ingrid.cfg = cfg

            if cfg.indir.path:
                # use input imagery
                ingrid = ingrid.set_indir()
            else:
                # set a source if specified or infer from location
                ingrid = ingrid.set_source()

            if cfg.outdir:
                ingrid = ingrid.set_outdir()

            # configure segmentation using cfg parameters
            ingrid = ingrid.set_segmentation()
            # configure vectorization using cfg parameters
            ingrid = ingrid.set_vectorization()

        return ingrid

    def to_cfg(
            self,
            file: Union[str, Path] = None
    ):
        raise NotImplementedError
        ...

    def to_pickle(
            self,
            path: Union[str, Path, None] = None
    ) -> Pickle:
        """
        Save InGrid instance to a pickle file for later reuse.

        Args:
            path: File path, directory path, or None for system temp directory

        Returns:
            Pickle object with path and MD5 hash

        Example:
            >>> ingrid: InGrid
            >>> pkl: Pickle = ingrid.to_pickle('/path/to/save.pkl')
        """

        cfg_hash = self.cfg.hash()
        filename = f'{self.location}.{cfg_hash}.pkl'

        if path is None:
            path = Path(tempfile.gettempdir()) / filename
        else:
            p = Path(path)
            path = p
            if not p.suffix:
                path = p / filename

        msg = f'Saving InGrid to pickle at \n\t{path}'
        logger.info(msg)

        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg.flatten()
        indir: Self = self.copy()
        indir.cfg = cfg

        with open(path, 'wb') as f:
            pickle.dump(indir, f)

        result = Pickle(path)
        return result

    @classmethod
    def from_pickle(
            cls,
            file: Union[
                str,
                Path,
            ]
    ) -> Self:
        """
        Load InGrid instance from a pickle file.

        Args:
            file: Path to pickle file

        Returns:
            InGrid instance

        Example:
            >>> ingrid: InGrid = InGrid.from_pickle('/path/to/save.pkl')
        """
        path = Path(file)
        logger.info(f'Loading InGrid from \n\t{path}')
        with open(path, 'rb') as f:
            # load the entire object using pickle
            instance = pickle.load(f)
        return instance

    @classmethod
    def from_basic(
            cls,
            outdir=None,
            location=None,
            pad=None,
            length=None
    ) -> Self:
        """
        Construct InGrid with basic configuration in one step.

        Args:
            outdir: Output directory path
            location: Location string (e.g., 'Boston Common, MA')
            pad: Padding pixels for segmentation
            length: vec-tile length

        Returns:
            Fully configured InGrid instance

        Example:
            >>> ingrid: InGrid = InGrid.from_basic(location='Boston, MA', pad=64)
        """

        ingrid = (
            InGrid
            .from_location(location)
            .set_source(
                outdir=outdir,
            )
            .set_segmentation(
                pad=pad,
            )
            .set_vectorization(
                length=length,
            )
        )
        return ingrid

    @property
    def ingrid(self) -> InGrid:
        """Quick access for the InGrid of a project."""
        return self

