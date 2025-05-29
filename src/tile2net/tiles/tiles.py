from __future__ import annotations
from requests.adapters import  HTTPAdapter

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from fiona.meta import extension
from functools import partial
from pathlib import Path
from typing import *

import PIL.Image
import certifi
import geopandas as gpd
import imageio.v3
import imageio.v3 as iio
import numpy as np
import pandas as pd
import pyproj
import requests
import shapely
from PIL import Image
from tqdm.auto import tqdm

from tile2net.logger import logger
from tile2net.raster import util
from . import util
from .explore import explore
from .fixed import GeoDataFrameFixed
from .indir import Indir
from .mosaic import Mosaic
from .source import Source
from .stitch import Stitch

if False:
    import folium
    from .stitched import Stitched


class Tiles(
    GeoDataFrameFixed,
):

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

    @Mosaic
    def mosaic(self):

        # This code block is just semantic sugar and does not run.
        # These columns are available once the tiles have been stitched:
        _ = (
            # xtile of the larger mosaic
            self.mosaic.xtile,
            # ytile of the larger mosaic
            self.mosaic.ytile,
            # row of the tile within the larger mosaic
            self.mosaic.r,
            # column of the tile within the larger mosaic
            self.mosaic.c,
        )

    @classmethod
    def from_bounds(
            cls,
            latlon: Union[
                str,
                list[float],
            ],
            zoom: int = 19,
    ) -> Self:
        """
        Create some base Tiles from bounds in lat, lon format and a
        slippy tile zoom level.
        """
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
    def from_integers(
            cls,
            tx,
            ty,
            zoom: int = 19,
    ) -> Self:
        """
        Construct Tiles from integer tile numbers.
        This allows for a non-rectangular grid of tiles.
        """
        if isinstance(ty, (pd.Series, pd.Index)):
            tn = ty.values
        elif isinstance(ty, np.ndarray):
            tn = ty
        else:
            raise TypeError('ty must be a Series, Index, or ndarray')
        tn = tn.astype('uint32')

        if isinstance(tx, (pd.Series, pd.Index)):
            tw = tx.values
        elif isinstance(tx, np.ndarray):
            tw = tx
        else:
            raise TypeError('tx must be a Series, Index, or ndarray')
        tw = tw.astype('uint32')

        te = tw + 1
        ts = tn + 1
        names = 'xtile ytile'.split()
        index = pd.MultiIndex.from_arrays([tx, ty], names=names)
        gw, gn = util.num2deg(tw, tn, zoom=zoom)
        ge, gs = util.num2deg(te, ts, zoom=zoom)
        trans = (
            pyproj.proj.Transformer
            .from_crs(4326, 3857, always_xy=True)
            .transform
        )
        pw, pn = trans(gw, gn)
        pe, ps = trans(ge, gs)
        geometry = shapely.box(pw, pn, pe, ps)

        result = cls(
            # data=data,
            geometry=geometry,
            index=index,
            # crs=4326,
            crs=3857,
        )
        result.zoom = zoom
        return result

    def with_indir(
            self,
            indir: str,
            extension: str = 'png',
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
        try:
            result.indir = indir
            result.extension = extension
        except ValueError as e:
            msg = (
                f'Invalid input directory: {indir}. '
                f'The directory directory must implicate the X and Y '
                f'tile numbers by including `x` and `y` in some format, '
                f'for example: '
                f'input/dir/x/y/z.png or input/dir/x_y_z.png.'
            )
            raise ValueError(msg) from e
        return result

    def with_outdir(
            self,
            outdir: Union[str, Path],
    ) -> Self:
        """
        Assign an output directory to the tiles.
        The tiles are saved to the output directory.
        """
        self.outdir = outdir

    def with_source(
            self,
            source=None,
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
            source = Source.from_inferred(source)
        result = self.copy()
        result.source = source
        result.source

        if indir:
            if Indir.is_valid(indir):
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
            args = (
                tempfile.gettempdir(),
                'tile2net',
                str(source.name),
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
            extension=source.extension,
        )

        urls = result.source.urls
        files = result.file
        self.download(files, urls, max_workers=max_workers)

        return result

    def download(
            self,
            paths: pd.Series,
            urls: pd.Series,
            retry: bool = True,
            force: bool = False,
            max_workers: int = 100,
    ) -> None:
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
        if paths.empty or urls.empty:
            return
        if len(paths) != len(urls):
            raise ValueError('paths and urls must be equal length')

            # ensure directories exist
        for p in paths.unique():
            Path(p).parent.mkdir(parents=True, exist_ok=True)

        paths_arr = paths.array
        urls_arr = urls.array

        if not force:
            exists_mask = np.fromiter(
                (Path(p).exists() for p in paths_arr),
                dtype=bool,
                count=len(paths_arr),
            )
            ndownload = exists_mask.__invert__().sum()
            msg = f'Downloading {ndownload:,} of {len(paths_arr):,} tiles.'
            logger.info(msg)
            if exists_mask.all():
                logger.info('All tiles already on disk.')
                return
            paths_arr = paths_arr[~exists_mask]
            urls_arr = urls_arr[~exists_mask]

        max_workers = min(max_workers, len(urls_arr))  # or higher if your system/network allows
        pool_size = max_workers  # keep-alive sockets you want

        with (
            ThreadPoolExecutor(max_workers=max_workers) as pool,
            requests.Session() as session
        ):
            session.mount(
                'https://',
                HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
            )
            session.headers.update({'User-Agent': 'tiles'})
            session.verify = certifi.where()

            def head(url: str) -> int:
                try:
                    return session.head(url, timeout=10).status_code
                except requests.exceptions.RequestException:
                    return -1

            msg = f'Checking {len(urls_arr):,} URLs...'
            logger.info(msg)

            it = tqdm(
                pool.map(head, urls_arr),
                total=len(urls_arr),
                desc='Checking status codes'
            )
            codes = np.fromiter(
                it,
                dtype=np.int16,
                count=len(urls_arr)
            )

            not_ok = codes != 200
            not_ok &= codes != 404
            if not_ok.any():
                logger.warning(f'Unexpected status codes: {np.unique(codes[not_ok])}')

            not_found = codes == 404
            if not_found.any():
                logger.info(f'{not_found.sum():,} tiles returned 404.')
                src = self.black.__fspath__()
                for p in paths_arr[not_found]:
                    os.symlink(src, p)
            paths_arr = paths_arr[~not_found]
            urls_arr = urls_arr[~not_found]

            if not len(paths_arr):
                return

            submit_get = partial(session.get, timeout=20)
            desc = f"Downloading {len(paths_arr):,} tiles..."
            it = tqdm(
                zip(urls_arr, paths_arr),
                total=len(paths_arr),
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
                writes.append(w)

            failed = [f.result() for f in as_completed(writes) if f.result()]

            if failed:
                if retry:
                    logger.error(f'{len(failed):,} failed tiles; retrying once.')
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

            logger.info('All requested tiles are on disk.')

    @property
    def xtile(self):
        """Tile integer X"""
        return self.index.get_level_values('xtile')

    @property
    def ytile(self):
        """Tile integer Y"""
        return self.index.get_level_values('ytile')

    @property
    def zoom(self) -> int:
        """Tile zoom level"""
        try:
            return self.attrs['zoom']
        except KeyError:
            raise NotImplementedError('write better warning')

    @zoom.setter
    def zoom(self, value: int):
        if not isinstance(value, int):
            raise TypeError('Zoom must be an integer')
        self.attrs['zoom'] = value

    @property
    def r(self) -> pd.Series:
        """Row of the tile within the overall grid."""
        ytile = (
            self.ytile
            .to_series()
            .set_axis(self.index)
        )
        result = ytile - ytile.min()
        return result

    @property
    def c(self) -> pd.Series:
        """Column of the tile within the overall grid."""
        xtile = (
            self.xtile
            .to_series()
            .set_axis(self.index)
        )
        result = xtile - xtile.min()
        return result

    @property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        if 'dimension' in self.attrs:
            return self.attrs['dimension']

        try:
            sample = next(p for p in self.file if Path(p).is_file())
        except StopIteration:
            raise FileNotFoundError('No image files found to infer dimension.')

        self.attrs['dimension'] = iio.imread(sample).shape[1]  # width
        return self.attrs['dimension']

    @dimension.setter
    def dimension(self, value: int):
        self.attrs['dimension'] = value

    @property
    def tscale(self) -> int:
        """
        Tile scale; the XYZ scale of the tiles.
        Higher value means smaller area.
        """
        return self.attrs.setdefault('tscale', self.zoom)

    @tscale.setter
    def tscale(self, value: int):
        if not isinstance(value, int):
            raise TypeError('Tile scale must be an integer')
        self.attrs['tscale'] = value

    @property
    def file(self) -> pd.Series:
        # try:
        #     return self['file']
        # except KeyError as e:
        #     # msg = (
        #     #     "Tiles.file has not been set. You must call "
        #     #     "`Tiles.with_indir()` or `Tiles.with_source()` to set the "
        #     #     "input directory or source for the tiles. The file column "
        #     #     "will then appear, which represents the file paths for the "
        #     #     "tiles on disk."
        #     # )
        #     # raise KeyError(msg) from e
        #     result = self.indir.files
        #     self['file'] = result

        if 'file' not in self:
            self['file'] = self.indir.files
        return self['file']

    @file.setter
    def file(self, value: pd.Series):
        self['file'] = value

    @property
    def outdir(self):
        try:
            return self.attrs['outdir']
        except KeyError as e:
            msg = (
                f'Tiles.outdir has not been set. '
                f'You must call `Tiles.with_outdir()` to set the output '
                f'directory for the tiles.'
            )
            raise ValueError(msg) from e

    # @property
    # def indir(self):
    #     try:
    #         return self.attrs['indir']
    #     except KeyError as e:
    #         msg = (
    #             f'Tiles.indir has not been set. '
    #             f'You must call `Tiles.with_indir()` to set the input '
    #             f'directory for the tiles.'
    #         )
    #         raise ValueError(msg) from e

    @property
    def stitched(self) -> Stitched:
        """If set, will return a Tiles DataFrame with stitched tiles."""
        if 'stitched' not in self.attrs:
            msg = (
                f'Tiles must be stitched using `Tiles.stitch` for '
                f'example `Tiles.stitch.to_resolution(2048)` or '
                f'`Tiles.stitch.to_cluster(16)`'
            )
            raise ValueError(msg)
        result = self.attrs['stitched']
        result.tiles = self
        return result

    @stitched.setter
    def stitched(self, value: Tiles):
        if not isinstance(value, Tiles):
            msg = """Tiles.stitched must be a Tiles object"""
            raise TypeError(msg)
        self.attrs['stitched'] = value

    def explore(
            self,
            *args,
            loc=None,
            tile='grey',
            subset='yellow',
            tiles: str = 'cartodbdark_matter',
            m=None,
            **kwargs
    ) -> folium.map:
        import folium
        if loc is None:
            m = explore(
                self,
                color=tile,
                name='lines',
                *args,
                **kwargs,
                tiles=tiles,
                m=m,
                style_kwds=dict(
                    # fill=False,
                    fillOpacity=0,
                )
            )
        else:
            tiles = self.loc[loc]
            loc = ~self.index.isin(tiles.index)
            m = explore(
                self.loc[loc],
                color=tile,
                name='tiles',
                *args,
                **kwargs,
                tiles=tiles,
                m=m,
                style_kwds=dict(
                    fill=False,
                    fillOpacity=0,
                )
            )
            m = explore(
                tiles,
                color=subset,
                name='subset',
                *args,
                **kwargs,
                tiles=tiles,
                m=m,
                style_kwds=dict(
                    fill=False,
                    fillOpacity=0,
                )
            )

        folium.LayerControl().add_to(m)
        return m

    # def view(
    #         self,
    #         maxdim: int = 2048,
    #         divider: Optional[str] = None,
    # ) -> PIL.Image:
    #     _ = imageio.v3.imread
    #     dimension = self.dimension
    #     shape = dimension, dimension, 3
    #     empty = np.zeros(shape, dtype=np.uint8)
    #     files: pd.Series = self.file
    #     R: pd.Series = self.r
    #     C: pd.Series = self.c

    def view(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
    ) -> PIL.Image.Image:
        print('⚠️AI GENERATED🤖')

        files: pd.Series = self.file
        R: pd.Series = self.r
        C: pd.Series = self.c
        dim = self.dimension

        th = self.dimension
        tw = self.dimension

        # scale factor to satisfy maxdim
        out_w, out_h = tw * dim, th * dim
        scale = 1.0
        if max(out_w, out_h) > maxdim:
            scale = maxdim / max(out_w, out_h)
            tw = max(1, int(round(tw * scale)))
            th = max(1, int(round(th * scale)))

        div_px = 1 if divider else 0
        full_w = dim * tw + div_px * (dim - 1)
        full_h = dim * th + div_px * (dim - 1)
        canvas_color = divider if divider else (0, 0, 0)
        out = Image.new('RGB', (full_w, full_h), color=canvas_color)

        def load(idx: int) -> tuple[int, int, np.ndarray]:
            arr = iio.imread(files.iat[idx])
            if scale != 1.0:
                arr = np.asarray(
                    Image.fromarray(arr).resize((tw, th), Image.Resampling.LANCZOS)
                )
            return R.iat[idx], C.iat[idx], arr

        with ThreadPoolExecutor() as exe:
            for r, c, arr in exe.map(load, range(len(files))):
                x = c * tw + div_px * c
                y = r * th + div_px * r
                out.paste(Image.fromarray(arr), (x, y))

        return out

    def view(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
    ) -> PIL.Image.Image:
        print('⚠️AI GENERATED🤖')

        files: pd.Series = self.file
        R: pd.Series = self.r        # 0-based row id
        C: pd.Series = self.c        # 0-based col id

        dim        = self.dimension  # original tile side length
        n_rows     = int(R.max()) + 1
        n_cols     = int(C.max()) + 1
        div_px     = 1 if divider else 0

        # full mosaic size before optional down-scaling
        full_w0    = n_cols * dim + div_px * (n_cols - 1)
        full_h0    = n_rows * dim + div_px * (n_rows - 1)

        scale      = 1.0
        if max(full_w0, full_h0) > maxdim:
            scale  = maxdim / max(full_w0, full_h0)

        tile_w     = max(1, int(round(dim * scale)))
        tile_h     = tile_w                                # square tiles
        full_w     = n_cols * tile_w + div_px * (n_cols - 1)
        full_h     = n_rows * tile_h + div_px * (n_rows - 1)

        canvas_col = divider if divider else (0, 0, 0)
        mosaic     = Image.new('RGB', (full_w, full_h), color=canvas_col)

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


    def __deepcopy__(self, memo) -> Tiles:
        return self.copy()


if __name__ == '__main__':
    ...
