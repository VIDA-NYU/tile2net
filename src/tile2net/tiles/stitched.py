from __future__ import annotations

import numpy as np
import pandas as pd

from .tiles import Tiles
from .outdir import Outdir


class Stitched(
    Tiles
):

    @property
    def tiles(self) -> Tiles:
        """Tiles object that this Stitched object is based on"""
        try:
            return self.attrs['tiles']
        except KeyError as e:
            raise AttributeError(

            ) from e

    @tiles.setter
    def tiles(self, value: Tiles):
        """Set the Tiles object that this Stitched object is based on"""
        if not isinstance(value, Tiles):
            raise TypeError(f"Expected Tiles object, got {type(value)}")
        self.attrs['tiles'] = value

    @property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        if 'dimension' in self.attrs:
            return self.attrs['dimension']

        tiles = self.tiles
        dscale = int(self.tscale - tiles.tscale)
        mlength = dscale ** 2
        result = tiles.dimension * mlength
        self.attrs['dimension'] = result
        return result

    @property
    def mlength(self):
        tiles = self.tiles
        dscale = int(self.tscale - tiles.tscale)
        return dscale ** 2

    @property
    def group(self) -> pd.Series:
        if 'group' in self.columns:
            return self['group']
        result = pd.Series(np.arange(len(self)), index=self.index)
        self['group'] = result
        return self['group']

