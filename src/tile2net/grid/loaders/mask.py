from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd

from .dataloader import DataLoader
from .dataset import DataSet
from .datawrapper import DataWrapper

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class MaskDataSet(
    DataSet
):
    @classmethod
    def from_tiles(
            cls,
            *,
            infile: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
    ) -> Self:
        wrapper = DataWrapper.from_tiles(
            infile=infile,
            index=index,
            row=row,
            col=col,
            background=background,
        )
        result = cls(wrapper)
        return result


class MaskDataLoader(
    DataLoader
):
    ...

