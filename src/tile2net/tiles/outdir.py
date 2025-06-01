from __future__ import annotations

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
    def files(self) -> pd.Series:
        tiles = self.tiles
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

    extension = 'shp'


class Network(
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

    extension = 'shp'


class Outdir(
    Dir
):
    @property
    def files(self):
        raise AttributeError

    @Polygons
    def polygons(self):
        ...

    @Network
    def network(self):
        ...

    @Probability
    def probability(self):
        ...

    @Error
    def error(self):
        ...
