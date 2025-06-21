from __future__ import annotations
import PIL.Image
import imageio.v3 as iio
import numpy as np
import pandas as pd
import pyproj
import shapely
from PIL import Image

import PIL.Image
import imageio.v3 as iio

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
import math
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import *

import imageio.v2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tile2net.tiles.dir.loader import Loader
from tile2net.tiles.cfg.logger import logger
from .predict import Predict
from .vectile import VecTile
from ..tiles import tile
from ..tiles.tiles import Tiles

if False:
    from ..intiles import InTiles


class Tile(
    tile.Tile
):
    tiles: SegTiles

    @tile.cached_property
    def length(self) -> int:
        """How many input tiles comprise a segmentation tile"""
        intiles = self.tiles.intiles
        result = 2 ** (intiles.tile.scale - self.scale)
        return result

    @tile.cached_property
    def dimension(self):
        """How many pixels in a inmentation tile"""
        segtiles = self.tiles
        intiles = segtiles.intiles
        result = intiles.tile.dimension * self.length
        return result


def __get__(
        self: SegTiles,
        instance: Tiles,
        owner: type[Tiles],
) -> SegTiles:
    if instance is None:
        return self
    try:
        result = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'InTiles must be segtiles using `InTiles.stitch` for '
            f'example `InTiles.stitch.to_resolution(2048)` or '
            f'`InTiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.intiles = instance
    return result


class SegTiles(
    Tiles,
):
    __name__ = 'segtiles'
    locals().update(
        __get__=__get__,
    )

    @tile.cached_property
    def intiles(self) -> InTiles:
        ...

    @Predict
    def predict(self):
        # This code block is just semantic sugar and does not run.
        # Take a look at the following methods which do run:
        result = (
            self.predict
            .with_polygons(
                max_hole_area=dict(
                    road=30,
                    crosswalk=15,
                ),
                grid_size=.001,
            )
            .to_outdir()
        )
        result = self.predict.to_outdir()

    def stitch(
            self
    ):
        intiles = self.intiles
        segtiles = self

        loc = ~intiles.segtile.skip
        infiles = intiles.infile.loc[loc]
        row = intiles.segtile.r.loc[loc]
        col = intiles.segtile.c.loc[loc]
        group = intiles.segtile.ipred.loc[loc]

        loc = ~segtiles.skip
        predfiles = segtiles.infile.loc[loc]
        n_missing = np.sum(loc)
        n_total = len(segtiles)

        if n_missing == 0:  # nothing to do
            msg = f'All {n_total:,} mosaics are already stitched.'
            logger.info(msg)
            return intiles
        else:
            logger.info(f'Stitching {n_missing:,} of {n_total:,} mosaics missing on disk.')

        loader = Loader(
            files=infiles,
            row=row,
            col=col,
            tile_shape=intiles.tile.shape,
            mosaic_shape=segtiles.tile.shape,
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
        it = zip(loader, predfiles)
        it = tqdm(it, 'stitching', n_missing, unit=' mosaic')

        writes = [
            executor.submit(imwrite, outfile, array)
            for array, outfile in it
        ]

        for w in writes:
            w.result()

        executor.shutdown(wait=True)

        del segtiles.skip
        assert segtiles.skip.all()

        return intiles

    @tile.cached_property
    def intiles(self) -> InTiles:
        """InTiles object that this SegTiles object is based on"""

    @property
    def ipred(self) -> pd.Series:
        key = 'ipred'
        if key in self.columns:
            return self[key]
        self[key] = np.arange(len(self), dtype=np.uint32)
        return self[key]

    @property
    def outdir(self):
        return self.intiles.outdir

    @property
    def cfg(self):
        return self.intiles.cfg

    @property
    def static(self):
        return self.intiles.static

    @VecTile
    def vectile(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the tiles have been stitched:
        # xtile of the vectile
        self.vectiles.xtile = ...
        # ytile of vectile
        self.vectiles.ytile = ...

    @Tile
    def tile(self):
        ...

    @property
    def infile(self) -> pd.Series:
        """
        A file for each segmentation tile: the stitched input tiles.
        Stitches input files when segtiles.file is accessed
        """
        key = 'infile'
        if key in self:
            return self[key]
        self[key] = self.intiles.outdir.segtiles.files()
        if not self.intiles.outdir.segtiles.skip().all():
            self.stitch()
        return self[key]

    @property
    def segtiles(self) -> Self:
        return self
    def preview(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
    ) -> PIL.Image.Image:

        files: pd.Series = self.infile
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
