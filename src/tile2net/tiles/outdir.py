from __future__ import annotations

import os

import datetime
import os.path

import pandas as pd

from .batchiterator import BatchIterator
from .dir import Dir


class Probability(
    Dir
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
        time = (
            datetime.datetime.now()
            .strftime('%d-%m-%Y_%H_%M')
        )
        os.makedirs(self.dir, exist_ok=True)
        file = os.path.join(self.dir, f'Polygons-{time}.parquet')
        cache[key] = file
        return file


class Network(
    Dir
):
    @property
    def file(self) -> str:
        key = f'{self._trace}.polygons'
        cache = self.tiles.attrs
        if key in cache:
            return cache[key]
        time = (
            datetime.datetime.now()
            .strftime('%d-%m-%Y_%H_%M')
        )
        os.makedirs(self.dir, exist_ok=True)
        file = os.path.join(self.dir, f'Network-{time}.parquet')
        return file


class SideBySide(
    Dir
):
    ...


class Outputs(
    Dir
):

    def files(self, dirname: str) -> pd.Series:
        tiles = self.tiles.stitched
        key = f'{self._trace}.{dirname}'
        if key in tiles:
            return tiles[key]
        suffix = (
            self.format
            .removeprefix(self.dir)
            .lstrip(os.sep)
        )
        format = os.path.join(self.dir, dirname, suffix)
        dir = os.path.dirname(format)
        os.makedirs(dir, exist_ok=True)
        zoom = tiles.zoom
        it = zip(tiles.ytile, tiles.xtile)
        data = [
            format.format(z=zoom, y=ytile, x=xtile)
            for ytile, xtile in it
        ]
        result = pd.Series(data, index=tiles.index)
        tiles[key] = result
        result = tiles[key]
        return result

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

    # def iterator(self, dirname, *args, **kwargs):
    #     key = f'{self._trace}.{dirname}.iterator'
    #     cache = self.tiles.attrs
    #
    #     try:  # → already cached ↴
    #         return cache[key]
    #     except KeyError:
    #         files = self.files(dirname, *args, **kwargs)
    #         if not self.tiles.cfg.force:  # drop skipped tiles
    #             files = files.loc[~self.tiles.outdir.skip]
    #
    #         return CachedIterator(self.tiles, files, cache, key)

    @BatchIterator
    def iterator(self, dirname: str):
        files = self.files(dirname)
        return files



class SegResults(
    Dir
):

    @Probability
    def prob(self):
        format = os.path.join(
            self.dir,
            'prob',
            self.suffix,
        ).replace(self.extension, 'png')
        result = Probability.from_format(format)
        return result

    @Error
    def error(self):
        format = os.path.join(
            self.dir,
            'error',
            self.suffix,
        ).replace(self.extension, 'png')
        result = Error.from_format(format)
        return result

    @SideBySide
    def sidebyside(self):
        format = os.path.join(
            self.dir,
            'sidebyside',
            self.suffix,
        ).replace(self.extension, 'png')
        result = SideBySide.from_format(format)
        return result

    @property
    def topn_failures(self) -> str:
        return os.path.join(self.dir, 'topn_failures.html')


class Submit(
    Dir
):
    ...


class MaskRaw(
    Dir
):
    ...


class Mask(
    Dir
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

    @Prediction
    def prediction(self):
        format = os.path.join(
            self.dir,
            'prediction',
            self.suffix,
        ).replace(self.extension, 'png')
        result = Prediction.from_format(format)
        return result

    @Outputs
    def outputs(self):
        format = os.path.join(
            self.dir,
            'outputs',
            self.suffix,
        )
        result = Outputs.from_format(format)
        return result

    @Mask
    def mask(self):
        format = os.path.join(
            self.dir,
            'mask',
            self.suffix,
        )
        result = Mask.from_format(format)
        return result

    @MaskRaw
    def raw(self):
        format = os.path.join(
            self.dir,
            'mask_raw',
            self.suffix,
        )
        result = MaskRaw.from_format(format)
        return result

    @SegResults
    def seg_results(self):
        format = os.path.join(
            self.dir,
            'seg_results',
            self.suffix,
        )
        result = SegResults.from_format(format)
        return result

    @Submit
    def submit(self):
        format = os.path.join(
            self.dir,
            'submit',
            self.suffix,
        )
        result = Submit.from_format(format)
        return result

    @Polygons
    def polygons(self):
        format = os.path.join(
            self.dir,
            'polygons',
            self.suffix,
        ).replace(self.extension, 'parquet')
        result = Polygons.from_format(format)
        return result

    @property
    def skip(self) -> pd.Series:
        tiles = self.tiles.stitched
        key = f'{self._trace}.skip'
        if key in tiles:
            return tiles[key]
        else:
            files: pd.Series = self.polygons.files()
            if self.tiles.cfg.force:
                data = False
            else:
                data = list(map(os.path.exists, files))
            result = pd.Series(data, index=tiles.index)
            tiles[key] = result
            return tiles[key]

    @Network
    def network(self):
        format = os.path.join(
            self.dir,
            'network',
            self.suffix,
        ).replace(self.extension, 'parquet')
        result = Network.from_format(format)
        return result

    @BestImages
    def best_images(self):
        format = os.path.join(
            self.dir,
            'best_images',
            self.suffix,
        )
        result = BestImages.from_format(format)
        return result

    def preview(self) -> str:
        self.seg_results.error.format
        self.seg_results.prob.format
        self.seg_results.sidebyside.format
        self.format
        self.mask.format
        self.mask.raw.format
        self.polygons.path
        self.network.path
