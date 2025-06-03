from __future__ import annotations

import datetime
import os.path

import pandas as pd

from .dir import Dir


class Probability(
    Dir
):
    @property
    def files(self) -> pd.Series:
        tiles = self.tiles.stitched
        key = self._trace
        if key in tiles:
            return tiles[key]
        else:
            format = self.format
            zoom = tiles.zoom
            it = zip(tiles.ytile, tiles.xtile)
            data = [
                format.format(z=zoom, y=ytile, x=xtile)
                for ytile, xtile in it
            ]
            result = pd.Series(data, index=tiles.index)
            tiles[key] = result
            return tiles[key]

    extension = 'npy'


class Error(
    Dir
):

    @property
    def files(self) -> pd.Series:
        tiles = self.tiles.stitched
        key = self._trace
        if key in tiles:
            return tiles[key]
        else:
            format = self.format
            zoom = tiles.zoom
            it = zip(tiles.ytile, tiles.xtile)
            data = [
                format.format(z=zoom, y=ytile, x=xtile)
                for ytile, xtile in it
            ]
            result = pd.Series(data, index=tiles.index)
            tiles[key] = result
            return tiles[key]

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
    @property
    def files(self) -> pd.Series:
        tiles = self.tiles.stitched
        key = self._trace
        if key in tiles:
            return tiles[key]
        else:
            format = self.format
            zoom = tiles.zoom
            it = zip(tiles.ytile, tiles.xtile)
            data = [
                format.format(z=zoom, y=ytile, x=xtile)
                for ytile, xtile in it
            ]
            result = pd.Series(data, index=tiles.index)
            tiles[key] = result
            return tiles[key]


class SegResults(
    Dir
):
    @Probability
    def prob(self):
        ...

    @Error
    def error(self):
        ...

    @SideBySide
    def sidebyside(self):
        ...

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
    @MaskRaw
    def raw(self):
        ...


class BestImages(
    Dir
):
    ...



class Outdir(
    Dir
):
    @SegResults
    def seg_results(self):
        ...

    @Submit
    def submit(self):
        ...

    @Polygons
    def polygons(self):
        ...

    @Network
    def network(self):
        ...

    @BestImages
    def best_images(self):
        ...


    def preview(self) -> str:
        self.seg_results.error.format
        self.seg_results.prob.format
        self.seg_results.sidebyside.format
        self.format
        self.mask.format
        self.mask.raw.format
        self.polygons.path
        self.network.path
