from functools import cached_property
from typing import *
from typing import Any, Union
from typing import TypeVar

import numpy as np
import pandas as pd

from .. import frame

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

    # @frame.column
    # def i(self):
    #     """mosaic identifiers; could be integer or destination path"""

    @frame.column
    def row(self):
        """row within the mosaic"""

    @frame.column
    def col(self):
        """column within the mosaic"""

    @cached_property
    def background(self):
        """background value to use for padding"""
        return 0

    @classmethod
    def from_tiles(
            cls,
            *,
            infile: ArrayLike,
            index: ArrayLike,
            row: ArrayLike,
            col: ArrayLike,
            background: int = None,
            force: bool | ArrayLike = True,
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
            infile=infile,
            row=row,
            col=col,
            force=force,
            **kwargs
        )
        frame = (
            pd.DataFrame(data)
            .set_axis(index)
        )
        names = frame.index.names
        by = [*names, 'row', 'col']
        frame = (
            frame
            .loc[lambda df: df.force]
            .reset_index()
            .sort_values(by=by)
            .set_index(names)
        )

        result: Self = cls.from_frame(frame)
        if background is not None:
            result.background = background
        return result
