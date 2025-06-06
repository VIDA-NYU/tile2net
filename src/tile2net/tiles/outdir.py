from __future__ import annotations

import datetime
import os.path

import pandas as pd

from .dir import Dir


class Probability(
    Dir
):
    extension = 'npy'


class Error(
    Dir
):


    extension = 'npy'


class Polygons(
    Dir
):
    @property
    def path(self) -> str:
        key = self._trace
        cache = self.tiles.attrs
        if key in cache:
            return cache[key]
        time = (
            datetime.datetime.now()
            .strftime('%d-%m-%Y_%H_%M')
        )
        path = os.path.join(self.dir, f'Network-{time}.shp')
        return path


class Network(
    Dir
):
    @property
    def path(self) -> str:
        key = self._trace
        cache = self.tiles.attrs
        if key in cache:
            return cache[key]
        time = (
            datetime.datetime.now()
            .strftime('%d-%m-%Y_%H_%M')
        )
        path = os.path.join(self.dir, f'Polygons-{time}.shp')
        return path


class SideBySide(
    Dir
):
    ...



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
        ).replace(self.extension, 'feather')
        result = Polygons.from_format(format)
        return result

    @Network
    def network(self):
        format = os.path.join(
            self.dir,
            'network',
            self.suffix,
        ).replace(self.extension, 'feather')
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
