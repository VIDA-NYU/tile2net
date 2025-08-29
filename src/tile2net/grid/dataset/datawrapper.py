
import copy

import torch
from typing import *
from functools import cached_property
import pandas as pd
import torch.utils.data
from .. import frame
from .dataset import DataSet
from typing import Any, Union
import pandas as pd
import numpy as np

from typing import TypeVar

T = TypeVar("T", bound="DataSet")
ArrayLike = Union[
    pd.Series,
    dict[Any, Any],
    list[Any],
    np.ndarray,
]


class DataWrapper(
    frame.FrameWrapper
):
    """
    Wrapper for a DataFrame which specifies the structure for the
    metadata of image tiles to be stitched into mosaics.
    """

    @frame.column
    def infile(self):
        """input file paths"""

    @frame.column
    def i(self):
        """mosaic identifiers; could be integer or destination path"""

    @frame.column
    def row(self):
        """row within the mosaic"""

    @frame.column
    def col(self):
        """column within the mosaic"""

    @classmethod
    def from_tiles(
            cls,
            *,
            infiles: ArrayLike,
            i: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = 0,
            **kwargs,
    ) -> Self:
        """
        infiles:
            series of input files
        i:
            identifier for each tile
        row:
            row index for each tile
        col:
            column index for each tile
        background:
            background value to use for padding
        """
        data = dict(
            infile=infiles,
            i=i,
            row=row,
            col=col,
            **kwargs
        )
        cols = 'i row col'.split()
        frame = (
            pd.DataFrame(data)
            .sort_values(by=cols)
        )
        result = cls.from_frame(frame)
        return result

