from __future__ import annotations
from tile2net.tiles.dir.loader import Loader
from tile2net.tiles.cfg.logger import logger
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


from ..dir import BatchIterator
from typing import *

import numpy as np
import pandas as pd
import rasterio

from ..tiles import tile

from ..tiles.tiles import Tiles

if False:
    from ..segtiles import SegTiles
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
        """How many input tiles comprise a segmentation tile"""
        vectiles = self.tiles
        intiles = vectiles.intiles
        result = 2 ** (intiles.tile.scale - self.scale) + 2 * self.padding
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
    def skip(self):
        key = 'skip'
        if key in self:
            return  self[key]
        self[key] = self.intiles.outdir.vectiles.skip()
        return self[key]

    @property
    def file(self):
        key = 'file'
        if key in self:
            return self[key]
        self[key] = self.intiles.outdir.vectiles.files()
        return self[key]

    @property
    def vectiles(self) -> Self:
        return self

    def stitch( self ):
        intiles = self.intiles
        segtiles = self.segtiles
        padded = segtiles.padded
        vectiles = self

        loc = ~padded.vectile.skip
        infiles = padded.infile.loc[loc]
        row = padded.vectile.r.loc[loc]
        col = padded.vectile.c.loc[loc]
        group = padded.vectile.ipred.loc[loc]

        loc = ~vectiles.skip
        predfiles = vectiles.infile.loc[loc]
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
        it = zip(loader, predfiles)
        it = tqdm(it, 'stitching', n_missing, unit=' mosaic')

        writes = [
            executor.submit(imwrite, outfile, array)
            for array, outfile in it
        ]

        for w in writes:
            w.result()

        executor.shutdown(wait=True)

        del vectiles.skip
        assert vectiles.skip.all()

        return padded

