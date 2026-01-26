from __future__ import annotations

from typing import *

from tile2net.grid.basegrid import file
from tile2net.grid.seggrid.postprocess import PostProcess

if TYPE_CHECKING:
    from tile2net.grid.seggrid import SegGrid


class DenseCRF(
    PostProcess
):
    instance: file.File
    basegrid: SegGrid
    """
    Namespace for work-in-progress postprocessing of segmentation results. 

    See usage:
        >>> SegGrid.file.postprocess
    """
