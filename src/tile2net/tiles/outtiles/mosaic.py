
from __future__ import annotations
from ..fixed import GeoDataFrameFixed
import pandas as pd

from typing import *

import numpy as np
from ..predtiles import PredTiles
from .. import predtiles
import pandas as pd
import rasterio

from .. import tile
from ..tiles import Tiles

class Mosaic(
    predtiles.Mosaic
):
    @property
    def r(self) -> pd.Series:
        """row within the mosaic of this tile"""
        predtiles = self.predtiles
        key = 'mosaic.r'
        if key in predtiles.columns:
            return predtiles[key]
        return result

    @property
    def c(self) -> pd.Series:
        """column within the mosaic of this tile"""
        predtiles = self.predtiles
        key = 'mosaic.c'
        if key in predtiles.columns:
            return predtiles[key]
        return result

