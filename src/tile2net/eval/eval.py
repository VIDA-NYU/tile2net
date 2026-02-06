from __future__ import annotations

from functools import *
from typing import *

import pandas as pd

from tile2net.eval.file import File
from tile2net.eval.ingrid import InGridNamespace as InGridNamespace
from tile2net.eval.seggrid import SegGridNamespace as SegGridNamespace
from tile2net.eval.stacked import Stacked
from tile2net.eval.vecgrid import VecGridNamespace as VecGridNamespace
from tile2net.core.dir.outdir import Outdir
from tile2net.core.frame.framewrapper import FrameWrapper

if TYPE_CHECKING:
    from tile2net.core.ingrid import InGrid


class Eval(
    FrameWrapper,
):
    grids: tuple[InGrid, ...]
    """Tuple of InGrids to compare."""

    @File
    def file(self):
        """
        Mirrors the `Grid.file` modules, but creates side-by-side comparisons
        of the respective files in the grids.
        """

    @Stacked
    def stacked(self):
        """Just like file, but the comparisons are stacked below the static images."""

    @cached_property
    def name(self) -> str:
        grid_names = [grid.name for grid in self.grids]
        return '_vs_'.join(grid_names)

    @classmethod
    def from_grids(
            cls,
            *grids: InGrid,
            outdir: str = None,
            name: str = None
    ) -> Self:
        if len(grids) < 2:
            raise ValueError("At least two grids are required for comparison")

        index = grids[0].index
        for grid in grids[1:]:
            index = index.intersection(grid.index)

        frame = pd.DataFrame(index=index)
        out = cls.from_frame(frame)

        if outdir is None:
            outdir = Outdir.from_parent(grids[0].outdir, 'eval')

        out.grids = grids
        out.outdir = outdir

        if name is not None:
            out.name = name

        return out


    @Outdir
    def outdir(self):
        ...

    @InGridNamespace
    def ingrid(self):
        ...

    @SegGridNamespace
    def seggrid(self):
        ...

    @VecGridNamespace
    def vecgrid(self):
        ...