from __future__ import annotations
from tile2net.tiles.dir.indir import Indir
from tile2net.tiles.dir.outdir import Outdir

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import *

import certifi
import geopandas as gpd
import imageio.v3
import imageio.v3
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm

from tile2net.tiles.logger import logger
from tile2net.raster import util
from tile2net.tiles.cfg import cfg
from .predtile import PredTile
from .stitch import Stitch
from .. import util
from ..cfg import Cfg
from tile2net.tiles.dir.dir import Dir
from ..predtiles import PredTiles
from .source import Source, SourceNotFound
from ..tiles import Tiles


class InTiles(
    Tiles
):

    @property
    def outtiles(self):
        return self.predtiles.outtiles

    @Stitch
    def stitch(self):
        # This code block is just semantic sugar and does not run.
        # Take a look at the following methods which do run:

        # stitch to a target resolution e.g. 2048 ptxels
        self.stitch.to_dimension(...)
        # stitch to a cluster size e.g. 16 tiles
        self.stitch.to_mosaic(...)
        # stitch to an XYZ scale e.g. 17
        self.stitch.to_scale(...)

    @PredTiles
    def predtiles(self):
        """
        After performing InTiles.stitch, InTiles.predtiles is
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
        self.with_source(...)  # automatically sets the source
        self.with_source('nyc')

    @Indir
    def indir(self):
        """
        Returns the Indir class, which wraps an input directory, for
        example `input/dir/x_y_z.png` or `input/dir/x/y/z.png`.
        See `Tiles.with_indir()` to actually set an input directory.
        """
        # This code block is just semantic sugar and does not run.
        # This method is how to set the input directory:
        self.with_indir(...)

    @Outdir
    def outdir(self):
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
        result = cls.from_integers(tx, ty, zoom=zoom)
        return result

    @classmethod
    def from_config(
            cls,
            args
    ):
        from tile2net.tiles.cfg import Cfg
        cfg = vars(args)
        cfg = Cfg(cfg)

        tiles = Tiles.from_location(
            location=cfg.location,
            zoom=cfg.zoom,
        )

        if cfg.source:
            # use specified source
            tiles = tiles.with_source(
                source=cfg.source,
                indir=cfg.input_dir,
            )
        elif cfg.input_dir:
            # use local files
            tiles = tiles.with_indir(
                indir=cfg.input_dir,
            )
        else:
            # infer source from location
            tiles = tiles.with_source(
                indir=cfg.input_dir,
            )

        pad = cfg.padding
        stitch = tiles.stitch
        if cfg.stitch.dimension:
            tiles = stitch.to_dimension(cfg.stitch.dimension, pad)
        elif cfg.stitch.mosaic:
            tiles = stitch.to_mosaic(cfg.stitch.mosaic, pad)
        elif cfg.stitch.scale:
            tiles = stitch.to_scale(cfg.stitch.scale, pad)

        tiles.predict(cfg.output_dir)
        raise NotImplemented

    @Cfg
    def cfg(self):
        # This code block is just semantic sugar and does not run.
        # You can access the various configuration options this way:
        _ = self.cfg.zoom
        _ = self.cfg.model.bs_val
        _ = self.cfg.polygon.max_hole_area
        # Please do not set the configuration options directly,
        # you may introduce bugs.

    @PredTile
    def predtile(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the tiles have been stitched:
        _ = (
            # xtile of the larger mosaic
            self.predtile.xtile,
            # ytile of the larger mosaic
            self.predtile.ytile,
            # row of the tile within the larger mosaic
            self.predtile.r,
            # column of the tile within the larger mosaic
            self.predtile.c,
        )

    def download(
            self,
            paths: pd.Series = None,
            urls: pd.Series = None,
            retry: bool = True,
            force: bool = False,
            max_workers: int = 100,
    ) -> Self:
        """
        infiles:
            Series of file path destinations
        urls:
            Series of URLs to download from
        retry:
            serialize tiles a second time if the first time fails
        force:
            redownload the tiles even if they already exist
        """
        if paths is None:
            paths = self.indir.files()
        if urls is None:
            urls = self.source.urls

        if paths.empty or urls.empty:
            return
        if len(paths) != len(urls):
            raise ValueError('paths and urls must be equal length')

            # ensure directories exist
        for p in paths.unique():
            Path(p).parent.mkdir(parents=True, exist_ok=True)

        PATHS = paths.array
        URLS = urls.array

        if not force:
            exists_mask = np.fromiter(
                (Path(p).exists() for p in PATHS),
                dtype=bool,
                count=len(PATHS),
            )
            if exists_mask.all():
                logger.info('All tiles already on disk.')
                return
            PATHS = PATHS[~exists_mask]
            URLS = URLS[~exists_mask]

        max_workers = min(max_workers, len(URLS))  # or higher if your system/network allows
        pool_size = max_workers  # keep-alive sockets you want

        with (
            ThreadPoolExecutor(max_workers=max_workers) as pool,
            requests.Session() as session
        ):

            adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
            session.mount('https://', adapter)
            session.headers.update({'User-Agent': 'tiles'})
            session.verify = certifi.where()

            def head(url: str) -> int:
                try:
                    return session.head(url, timeout=10).status_code
                except requests.exceptions.RequestException:
                    return -1

            # Only check URLs whose files do not already exist
            it = map(os.path.exists, PATHS)
            exists = np.fromiter(it, bool, count=len(PATHS))
            if exists.all():
                return

            PATHS = PATHS[~exists]
            URLS = URLS[~exists]

            if len(URLS):
                msg = f'Checking {len(URLS):,} URLs...'
                logger.info(msg)

            it = tqdm(
                pool.map(head, URLS),
                total=len(URLS),
                desc='Checking status codes'
            )
            codes = np.fromiter(
                it,
                dtype=np.int16,
                count=len(URLS)
            )

            # not_ok = (codes != 200) & (codes != 404)
            not_ok = codes != 200
            not_ok &= codes != 404
            if np.any(not_ok):
                logger.warning(f'Unexpected status codes: {np.unique(codes[not_ok])}')

            not_found = codes == 404

            n = np.sum(not_found)
            N = len(PATHS)
            if np.any(not_found):
                if n == N:
                    msg = f'All {n:,} requested tiles returned 404.'
                    raise FileNotFoundError(msg)
                msg = f'{n:,} out of {N:,} tiles returned 404.'
                logger.warning(msg)

            paths = PATHS[~not_found]
            urls = URLS[~not_found]

            if not len(paths):
                return

            submit_get = partial(session.get, timeout=20)
            desc = f"Downloading {len(paths):,} tiles to {self.indir.format}"
            it = tqdm(
                zip(urls, paths),
                total=len(paths),
                desc=desc
            )

            futures = {
                pool.submit(submit_get, url): Path(path)
                for url, path in it
            }

            def write(path: Path, data: bytes) -> Path | None:
                path.write_bytes(data)
                try:
                    imageio.v3.imread(path)
                except ValueError:
                    path.unlink(missing_ok=True)
                    return path
                return None

            writes = []
            for fut in tqdm(as_completed(futures), total=len(futures), desc='Writing...'):
                resp = fut.result()
                path = futures[fut]
                try:
                    resp.raise_for_status()
                except requests.exceptions.HTTPError:
                    continue
                w = pool.submit(write, path, resp.content)
            failed = [
                f.result()
                for f in as_completed(writes)
                if f.result()
            ]

            if failed:
                if retry:
                    msg = f'{len(failed):,} failed tiles; retrying once.'
                    logger.error(msg)
                    self.download(
                        paths=pd.Series(failed, dtype=object),
                        urls=pd.Series(
                            [urls.iloc[paths.eq(p).argmax()] for p in failed],
                            dtype=object,
                        ),
                        retry=False,
                        force=force,
                    )
                else:
                    raise FileNotFoundError(f'{len(failed):,} tiles failed: {failed[:3]}')

            def _link_or_copy(src, dst):
                try:
                    os.symlink(src, dst)
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)

            paths = PATHS[not_found]
            black = self.static.black

            if len(paths):
                msg = f'Linking or copying {len(paths):,} tiles to {black}.'
                logger.info(msg)

            if len(paths) > 64:  # thread only when worthwhile
                with ThreadPoolExecutor() as ex:
                    ex.map(lambda p: _link_or_copy(black, p), paths)
            else:
                for path in paths:
                    _link_or_copy(black, path)

            logger.info('All requested tiles are on disk.')

        return self

    def with_source(
            self,
            source=None,
            name: str = None,
            indir: Union[str, Path] = None,
            max_workers: int = 100,
    ) -> Self:
        """
        Assign a source to the tiles. The tiles are downloaded from
        the source and saved to an input directory.

        You can pass an indir which will be the destination the files
        are saved to.
        """
        if source is None:
            source = (
                gpd.GeoSeries([self.union_all()], crs=self.crs)
                .to_crs(4326)
                .pipe(Source.from_inferred)
            )
        elif isinstance(source, Source):
            ...
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
        if name:
            result.name = name
        if indir:
            if Dir.is_valid(indir):
                ...
            else:
                args = (
                    indir,
                    f'x_y_z.{source.extension}',
                )
                indir = (
                    Path(*args)
                    .resolve()
                    .__str__()
                )
        else:
            if result.name:
                name = result.name
            else:
                name = result.source.name
            args = (
                tempfile.gettempdir(),
                'tile2net',
                name,
                'z',
                'x_y.png'
            )
            indir = (
                Path(*args)
                .resolve()
                .__str__()
            )
        result = result.with_indir(
            indir,
            name=name,
        )

        # urls = result.source.urls
        # files = result.indir.files()
        # result.download(files, urls, max_workers=max_workers)
        result.download(max_workers=max_workers)

        return result

    # def skip(self):
    #     loc = self.indir.files()
    def with_indir(
            self,
            indir: str,
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
        try:
            _ = result.outdir
        except AttributeError:
            msg = (
                f'Output directory not yet set. Based on the input directory, '
                f'setting it to a default value.'
            )
            logger.info(msg)
            result = result.with_outdir()
        return result

    def with_outdir(
            self,
            outdir: Union[str, Path] = None,
    ) -> Self:
        """
        Assign an output directory to the tiles.
        The tiles are saved to the output directory.
        """
        result: Tiles = self.copy()
        if outdir:
            ...
        elif cfg.output_dir:
            outdir = cfg.output_dir
        else:
            if self.name:
                name = self.name
            elif cfg.name:
                name = cfg.name
            elif (
                    self.source
                    and self.source.name
            ):
                name = self.source.name
            else:
                msg = (
                    f'No output directory specified, and unable to infer '
                    f'a name for a temporary output directory. Either '
                    f'set a name, use a source, or specify an output directory.'
                )
                raise ValueError(msg)

            outdir = os.path.join(
                tempfile.gettempdir(),
                'tile2net',
                name,
                'outdir'
            )

        try:
            result.outdir = outdir
        except ValueError:
            retry = os.path.join(outdir, 'z', 'x_y.png')
            result.outdir = retry
            logger.info(f'Setting output directory to \n\t{retry}')
        else:
            logger.info(f'Setting output directory to \n\t{outdir}')

        return result

