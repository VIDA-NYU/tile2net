from __future__ import annotations
from .intiles import InTiles

from .vectiles import VecTiles
from .segtiles import SegTiles

import os
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import os

import datetime
import os.path

import pandas as pd
from pandas.tseries.holiday import USPresidentsDay

from .batchiterator import BatchIterator
from .dir import Dir, Dir, Dir, Dir

if False:
    import tile2net.tiles.intiles



class Probability(
    Dir,
):
    extension = 'npy'


class Prediction(
    Dir
):
    extension = 'png'


class Error(
    Dir
):
    extension = 'npy'


class Polygons(
    Dir
):

    @property
    def file(self) -> str:
        key = f'{self._trace}.polygons'
        cache = self.tiles.attrs

        if key in cache:
            return cache[key]

        # ───── deterministic UUID from tile indices ───── #
        # unique (x, y) pairs ensure ordering does not alter digest
        stack = [self.tiles.xtile.to_numpy(), self.tiles.ytile.to_numpy()]
        pairs = np.unique(
            np.column_stack(stack),
            axis=0,
        )
        digest = hashlib.blake2b(pairs.tobytes(), digest_size=8).hexdigest()  # 16 hex

        Path(self.dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.dir, f'Polygons-{digest}.parquet')
        cache[key] = filename
        return filename


class Lines(
    Dir
):

    @property
    def file(self) -> str:
        key = f'{self._trace}.lines'
        cache = self.tiles.attrs
        if key in cache:
            return cache[key]

        pairs = np.unique(
            np.column_stack([self.tiles.xtile.to_numpy(), self.tiles.ytile.to_numpy()]),
            axis=0,
        )
        digest = hashlib.blake2b(pairs.tobytes(), digest_size=8).hexdigest()  # 16-hex UUID

        Path(self.dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.dir, f'Lines-{digest}.parquet')
        cache[key] = filename
        return filename


class SideBySide(
    Dir
):
    ...


class Outputs(
    Dir,
):
    ...

    # def iterator(self, dirname: str, *args, **kwargs) -> Iterator[pd.Series]:
    #     return super(Outputs, self).iterator(dirname)
    #     key = self._trace
    #     cache = self.tiles.attrs
    #     if key in cache:
    #         it = cache[key]
    #     else:
    #         files = self.files(dirname)
    #         if not self.tiles.cfg.force:
    #             loc = ~self.tiles.outdir.skip
    #             files = files.loc[loc]
    #         it = iter(files)
    #         cache[key] = it
    #     yield from it
    #     del cache[key]
    #


# class SegResults(
#     Dir,
# ):
#
#     @Probability
#     def prob(self):
#         format = os.path.join(
#             self.dir,
#             'prob',
#             self.suffix,
#         ).replace(self.extension, 'png')
#         result = Probability.from_format(format)
#         return result
#
#     @Error
#     def error(self):
#         format = os.path.join(
#             self.dir,
#             'error',
#             self.suffix,
#         ).replace(self.extension, 'png')
#         result = Error.from_format(format)
#         return result
#
#     @SideBySide
#     def sidebyside(self):
#         format = os.path.join(
#             self.dir,
#             'sidebyside',
#             self.suffix,
#         ).replace(self.extension, 'png')
#         result = SideBySide.from_format(format)
#         return result
#
#     @property
#     def topn_failures(self) -> str:
#         return os.path.join(self.dir, 'topn_failures.html')
#

class Submit(
    Dir,
):
    ...


class MaskRaw(
    Dir,
):
    ...


class Mask(
    Dir,
):
    ...


class BestImages(
    Dir
):
    @property
    def webpage(self):
        return os.path.join(self.dir, 'webpage.html')


class Outdir(
    Dir
):
    tiles: tile2net.tiles.intiles.InTiles

    # @Outputs
    # def outputs(self):
    #     format = os.path.join(
    #         self.dir,
    #         'outputs',
    #         self.suffix,
    #     )
    #     result = Outputs.from_format(format)
    #     return result
    # @Submit
    # def submit(self):
    #     format = os.path.join(
    #         self.dir,
    #         'submit',
    #         self.suffix,
    #     )
    #     result = Submit.from_format(format)
    #     return result
    #
    #
    # @BestImages
    # def best_images(self):
    #     format = os.path.join(
    #         self.dir,
    #         'best_images',
    #         self.suffix,
    #     )
    #     result = BestImages.from_format(format)
    #     return result

    @VecTiles
    def vectiles(self):
        ...

        # format = os.path.join(
        #     self.dir,
        #     'vectiles',
        #     self.suffix
        # ).replace(self.extension, 'png')
        # result = VecTiles.from_format(format)
        # return result

    @SegTiles
    def segtiles(self):
        ...

        # format = os.path.join(
        #     self.dir,
        #     'segtiles',
        #     self.suffix
        # ).replace(self.extension, 'png')
        # result = SegTiles.from_format(format)
        # return result

    @InTiles
    def intiles(self):
        ...

        # format = os.path.join(
        #     self.dir,
        #     'intiles',
        #     self.suffix
        # ).replace(self.extension, self.intiles.indir.extension)
        # result = InTiles.from_format(format)
        # return result


    @Lines
    def lines(self):
        ...

    @Polygons
    def polygons(self):
        ...





    # def preview(self) -> str:
    #     self.seg_results.error.format
    #     self.seg_results.prob.format
    #     self.seg_results.sidebyside.format
    #     self.format
    #     self.mask.format
    #     self.mask.raw.format
    #     self.polygons.path
    #     self.network.path
