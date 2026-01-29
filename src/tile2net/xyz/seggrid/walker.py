from __future__ import annotations

import os
from typing import *

import pandas as pd

from tile2net.xyz import frame
from tile2net.xyz.basegrid import file
from tile2net.xyz.seggrid.postprocess import PostProcess

if TYPE_CHECKING:
    from tile2net.xyz.seggrid import SegGrid


class Walker(
    PostProcess
):
    instance: file.File
    basegrid: SegGrid
    """
    Namespace for work-in-progress postprocessing of segmentation results. 

    See usage:
        >>> SegGrid.file.postprocess
    """

    @frame.column
    def prob(self) -> pd.Series:
        """Segmentation masks, where each pixel is a class id"""
        grid = self.basegrid
        probs = grid.file.prob
        files: pd.Series = (
            self.basegrid.outdir.seggrid
            .__getattribute__(self.__name__)
            .files(self.basegrid)
        )
        if not files.map(os.path.exists).all():
            ...
        return files
