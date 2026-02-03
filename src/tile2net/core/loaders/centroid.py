from __future__ import annotations

from functools import *
from typing import *

import numpy as np
import pandas as pd

from .stitch import T

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]
from .mask import MaskDataSet, MaskDataLoader


class CentroidDataSet(
    MaskDataSet
):

    def __getitem__(self, item):
        result = super().__getitem__(item)

    @cached_property
    def coarse(self):
        return False

    @cached_property
    def custom_coarse(self):
        return False


class CentroidDataLoader(
    MaskDataLoader
):
    ...
