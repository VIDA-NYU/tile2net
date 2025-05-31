from __future__ import annotations

import numpy as np
import pandas as pd

from .tiles import Tiles

class Predictions(
    Tiles
):
    ...

    @property
    def outdir(self):
        ...

    @property
    def probability(self) -> pd.Series:
        ...

    @property
    def error(self) -> pd.Series:
        ...

    @property
    def file(self) -> pd.Series:
        if 'file' in self.columns:
            return self['file']
        self.outdir.files

    @property
    def polygons(self):
        ...

    @property
    def network(self):
        ...


