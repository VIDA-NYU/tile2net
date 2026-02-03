from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd

from .dataloader import DataLoader
from .stitch import StitchDataSet
from .datawrapper import DataWrapper

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class ImageDataSet(
    StitchDataSet
):

    @classmethod
    def from_columns(
            cls,
            *,
            static: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
            **kwargs,
    ) -> Self:
        wrapper = DataWrapper.from_columns(
            image_path=static,
            index=index,
            row=row,
            col=col,
            background=background,
            **kwargs
        )
        result = cls(wrapper)
        return result


class StaticDataLoader(
    DataLoader
):
    ...
