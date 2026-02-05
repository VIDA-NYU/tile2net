from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd

from tile2net.core.loaders.stream import StreamStitchDataSet
from .dataloader import DataLoader
from .stitch import StitchDataSet

ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class ImageDataSet(
    StitchDataSet
):

    ...



class StaticDataLoader(
    DataLoader
):
    ...


class StreamImageDataSet(
    StreamStitchDataSet,
    ImageDataSet,
):
    ...
