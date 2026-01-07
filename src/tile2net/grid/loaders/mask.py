from __future__ import annotations
import torch
from functools import *

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


class MaskDataSet(
    StitchDataSet
):
    @classmethod
    def from_tiles(
            cls,
            *,
            static: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
    ) -> Self:
        wrapper = DataWrapper.from_tiles(
            static=static,
            index=index,
            row=row,
            col=col,
            background=background,
        )
        result = cls(wrapper)
        return result

    @cached_property
    def give_placeholders(self) -> bool:
        """
        If the mask paths are provided, we load them normally.
        If no mask paths are provided, we return -1 to save memory.
        If some mask paths are provided, we have an unexpected situation.
        """
        isna = self.wrapper.static.isna()
        if not isna.any():
            return False
        elif isna.all():
            return True
        else:
            msg = f'Mask files must be all or nothing: only some were missing files.'
            raise ValueError(msg)

    def __getitem__(self, item) -> Union[int, np.ndarray]:
        if self.give_placeholders:
            return -1
        return super().__getitem__(item)


class MaskDataLoader(
    DataLoader
):
    ...
