import pandas as pd
from .source import Source
from .dir import Dir

class Indir(
    Dir
):
    def files(self) -> pd.Series:
        tiles = self.tiles
        key = f'dir.{self.__name__}'
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
