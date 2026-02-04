from __future__ import annotations

import os
from typing import *

import pandas as pd

from tile2net.core import frame
from tile2net.core.grid import file
from tile2net.core.seggrid.postprocess import PostProcess

if TYPE_CHECKING:
    from tile2net.core.seggrid import SegGrid


class Walker(
    PostProcess
):
    instance: file.File
    grid: SegGrid
    """
    Namespace for work-in-progress postprocessing of segmentation results. 

    See usage:
        >>> SegGrid.file.postprocess
    """

    @frame.column
    def prob(self) -> pd.Series:
        """Segmentation masks, where each pixel is a class id"""
        grid = self.grid
        probs = grid.file.prob
        files: pd.Series = (
            self.grid.outdir.seggrid
            .__getattribute__(self.__name__)
            .files(self.grid)
        )
        if not files.map(os.path.exists).all():
            ...
        return files
