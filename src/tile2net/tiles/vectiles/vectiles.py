from __future__ import annotations

import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pathlib import Path
from typing import *

import PIL.Image
import PIL.Image
import imageio.v2
import imageio.v3
import imageio.v3
import imageio.v3 as iio
import numpy
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import tile2net.tiles.tileseg.network.ocrnet
from tile2net.tiles.cfg.cfg import assert_and_infer_cfg
from tile2net.tiles.cfg.logger import logger
from tile2net.tiles.dir.loader import Loader
from tile2net.tiles.tiles.static import Static
from tile2net.tiles.tileseg import datasets
from tile2net.tiles.tileseg import network
from tile2net.tiles.tileseg.loss.optimizer import get_optimizer, restore_opt, restore_net
from tile2net.tiles.tileseg.loss.utils import get_loss
from tile2net.tiles.tileseg.network.ocrnet import MscaleOCR
from tile2net.tiles.tileseg.utils.misc import AverageMeter, prep_experiment
from ..tiles import file
from ..tiles import tile
from ..tiles.tiles import Tiles
from ...tiles.util import recursion_block

import os.path
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import *

import imageio.v2
import imageio.v3
import imageio.v3
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from tqdm.auto import tqdm

from tile2net.tiles.cfg.logger import logger
from tile2net.tiles.dir.loader import Loader
from ..dir import BatchIterator
from ..tiles import file
from ..tiles import tile
from ..tiles.tiles import Tiles
from ...tiles.util import recursion_block

if False:
    from ..intiles import InTiles


def __get__(
        self: Padding,
        instance: VecTiles,
        owner,
) -> Padding:
    self.vectiles = instance
    return self


class Padding(

):
    vectiles: VecTiles = None
    locals().update(
        __get__=__get__,
    )

    @property
    def gw(self) -> pd.Series:
        vectiles = self.vectiles
        padded = vectiles.padded
        haystack = padded.vectile.index
        index = vectiles.index
        top_left = np.zeros_like((len(index), 2))
        needles = pd.MultiIndex.append(index, top_left)
        result = (
            padded
            .set_axis(haystack)
            .loc[needles, 'gw']
            .values
        )
        raise NotImplementedError

    @property
    def gn(self) -> pd.Series:
        ...

    @property
    def ge(self) -> pd.Series:
        ...

    @property
    def gs(self) -> pd.Series:
        ...


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
        files = tiles.intiles.outdir.segtiles.files(tiles)
        if (
            not tiles.stitch
            and not files.map(os.path.exists).all()
        ):
            tiles.stitch()
        tiles[key] = files
        return tiles[key]

    @property
    def network(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.network'
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
    def affine_params(self) -> pd.Series:
        key = 'affine_params'
        if key in self:
            return self[key]

        dim = self.tile.dimension
        self: pd.DataFrame
        col = 'gw gs ge gn'.split()
        it = self[col].itertuples(index=False)
        data = [
            rasterio.transform
            .from_bounds(gw, gs, ge, gn, dim, dim)
            for gw, gs, ge, gn in it
        ]
        result = pd.Series(data, index=self.index, name=key)
        self[key] = result
        return self[key]

    @BatchIterator
    def affine_iterator(self):
        return self.affine_params

    @property
    def vectiles(self) -> Self:
        return self

    @recursion_block
    def stitch(self):
        segtiles = self.segtiles
        vectiles = self
        padded = segtiles.padded
        loc = ~padded.vectile.stitched.map(os.path.exists)
        infiles = padded.file.maskraw.loc[loc]
        row = padded.vectile.r.loc[loc]
        col = padded.vectile.c.loc[loc]
        group = padded.vectile.stitched.loc[loc]

        loc = ~vectiles.file.stitched.map(os.path.exists)
        predfiles = vectiles.file.stitched.loc[loc]
        n_missing = np.sum(loc)
        n_total = len(vectiles)

        if n_missing == 0:  # nothing to do
            msg = f'All {n_total:,} mosaics are already stitched.'
            logger.info(msg)
            return padded
        else:
            logger.info(f'Stitching {n_missing:,} of {n_total:,} mosaics missing on disk.')

        loader = Loader(
            files=infiles,
            row=row,
            col=col,
            tile_shape=padded.tile.shape,
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
        return padded


    @recursion_block
    def vectorize(self):
        ...

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
