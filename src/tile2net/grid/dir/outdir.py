from __future__ import annotations

import hashlib
import os
import os.path
from pathlib import Path

import numpy as np

from .dir import Dir
from .ingrid import InGrid
from .seggrid import SegGrid
from .vecgrid import VecGrid

if False:
    import tile2net.grid.ingrid



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
        cache = self.grid.__dict__

        if key in cache:
            return cache[key]

        # ───── deterministic UUID from tile indices ───── #
        # unique (x, y) pairs ensure ordering does not alter digest
        stack = [self.grid.xtile.to_numpy(), self.grid.ytile.to_numpy()]
        pairs = np.unique(
            np.column_stack(stack),
            axis=0,
        )
        digest = hashlib.blake2b(pairs.tobytes(), digest_size=8).hexdigest()  # 16 hex

        Path(self.dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.dir, f'Polygons-{digest}.parquet')
        cache[key] = filename
        return filename
        
    @property
    def preview(self) -> str:
        key = f'{self._trace}.preview'
        cache = self.grid.__dict__
        
        if key in cache:
            return cache[key]
            
        # ───── deterministic UUID from tile indices ───── #
        # unique (x, y) pairs ensure ordering does not alter digest
        stack = [self.grid.xtile.to_numpy(), self.grid.ytile.to_numpy()]
        pairs = np.unique(
            np.column_stack(stack),
            axis=0,
        )
        digest = hashlib.blake2b(pairs.tobytes(), digest_size=8).hexdigest()  # 16 hex
        
        Path(os.path.join(self.dir, 'preview')).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.dir, 'preview', f'Polygons-{digest}.png')
        cache[key] = filename
        return filename


class Lines(
    Dir
):

    @property
    def file(self) -> str:
        key = f'{self._trace}.lines'
        cache = self.grid.__dict__
        if key in cache:
            return cache[key]

        pairs = np.unique(
            np.column_stack([self.grid.xtile.to_numpy(), self.grid.ytile.to_numpy()]),
            axis=0,
        )
        digest = hashlib.blake2b(pairs.tobytes(), digest_size=8).hexdigest()  # 16-hex UUID

        Path(self.dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.dir, f'Lines-{digest}.parquet')
        cache[key] = filename
        return filename
        
    @property
    def preview(self) -> str:
        key = f'{self._trace}.preview'
        cache = self.grid.__dict__
        if key in cache:
            return cache[key]
            
        pairs = np.unique(
            np.column_stack([self.grid.xtile.to_numpy(), self.grid.ytile.to_numpy()]),
            axis=0,
        )
        digest = hashlib.blake2b(pairs.tobytes(), digest_size=8).hexdigest()  # 16-hex UUID
        
        Path(os.path.join(self.dir, 'preview')).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.dir, 'preview', f'Lines-{digest}.png')
        cache[key] = filename
        return filename


class SideBySide(
    Dir
):
    ...



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
    grid: tile2net.grid.ingrid.InGrid

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

    @VecGrid
    def vecgrid(self):
        format = os.path.join(
            self.dir,
            'vecgrid',
            self.suffix
        )
        result = VecGrid.from_format(format)
        return result

    @SegGrid
    def seggrid(self):
        format = os.path.join(
            self.dir,
            'seggrid',
            self.suffix
        )
        result = SegGrid.from_format(format)
        return result

    @InGrid
    def ingrid(self):
        format = os.path.join(
            self.dir,
            'ingrid',
            self.suffix
        )
        result = InGrid.from_format(format)
        return result


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
