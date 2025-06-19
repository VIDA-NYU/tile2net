from __future__ import annotations


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
from .dir import Dir, UsesInTiles, UsesOutTiles, UsesPredTiles



class Probability(
    UsesPredTiles,
):
    extension = 'npy'


class Prediction(
    UsesPredTiles
):
    extension = 'png'


class Error(
    UsesPredTiles
):
    extension = 'npy'



class Polygons(
    UsesOutTiles
):
    @property
    def tiles(self):
        return self.intiles.outtiles

    @property
    def file(self) -> str:
        key = f'{self._trace}.polygons'
        cache = self.intiles.attrs

        if key in cache:
            return cache[key]

        # ───── deterministic UUID from tile indices ───── #
        # unique (x, y) pairs ensure ordering does not alter digest
        pairs = np.unique(
            np.column_stack([self.intiles.xtile.to_numpy(), self.intiles.ytile.to_numpy()]),
            axis=0,
        )
        digest = hashlib.blake2b(pairs.tobytes(), digest_size=8).hexdigest()  # 16 hex

        Path(self.dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.dir, f'Polygons-{digest}.parquet')
        cache[key] = filename
        return filename


class Network(
    UsesOutTiles
):
    @property
    def tiles(self):
        return self.intiles.outtiles

    @property
    def file(self) -> str:
        key = f'{self._trace}.network'
        cache = self.intiles.attrs
        if key in cache:
            return cache[key]

        pairs = np.unique(
            np.column_stack([self.intiles.xtile.to_numpy(), self.intiles.ytile.to_numpy()]),
            axis=0,
        )
        digest = hashlib.blake2b(pairs.tobytes(), digest_size=8).hexdigest()  # 16-hex UUID

        Path(self.dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.dir, f'Network-{digest}.parquet')
        cache[key] = filename
        return filename


class SideBySide(
    UsesPredTiles
):
    ...


class Outputs(
    UsesPredTiles,
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




class SegResults(
    UsesPredTiles,
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
    UsesPredTiles,
):
    ...


class MaskRaw(
    UsesPredTiles,
):
    ...


class Mask(
    UsesPredTiles,
):
    ...


class OutTiles(
    UsesOutTiles
):
    ...

class PredTiles(
    UsesPredTiles,
):
    ...


class BestImages(
    UsesPredTiles
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
        tiles = self.intiles.predtiles
        key = f'{self._trace}.skip'
        if key in tiles:
            return tiles[key]
        else:
            files: pd.Series = self.polygons.files()
            if self.intiles.cfg.force:
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

    @OutTiles
    def outtiles(self):
        format = os.path.join(
            self.dir,
            'outtiles',
            self.suffix
        ).replace(self.extension, 'png')
        result = OutTiles.from_format(format)
        return result

    @PredTiles
    def predtiles(self):
        format = os.path.join(
            self.dir,
            'predtiles',
            self.suffix
        ).replace(self.extension, 'png')
        result = PredTiles.from_format(format)
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
