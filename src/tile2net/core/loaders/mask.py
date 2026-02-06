from __future__ import annotations

from functools import *
from typing import *

import numpy as np
import pandas as pd

from .dataloader import DataLoader
from .stitch import StitchDataSet
from .stream import StreamStitchDataSet

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class MaskDataSet(
    StitchDataSet
):
    @cached_property
    def give_placeholders(self) -> bool:
        """
        If the mask paths are provided, we load them normally.
        If no mask paths are provided, we return -1 to save memory.
        If some mask paths are provided, we have an unexpected situation.
        """
        isna = self.wrapper.image_paths.isna()
        if isna.all():
            return True
        elif isna.any():
            msg = f'Mask files must be all or nothing: only some were missing files.'
            raise ValueError(msg)
        else:
            return False

    def __getitem__(self, item) -> Union[int, np.ndarray]:
        if self.give_placeholders:
            return -1
        return super().__getitem__(item)


class MaskDataLoader(
    DataLoader
):
    ...


class StreamMaskDataSet(
    StreamStitchDataSet,
    MaskDataSet,
):
    ...



