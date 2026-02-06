from __future__ import annotations

from functools import *
from typing import *

import pandas as pd

from tile2net.core.compare.file import File
from tile2net.core.compare.ingrid import InGridNamespace as InGridNamespace
from tile2net.core.compare.seggrid import SegGridNamespace as SegGridNamespace
from tile2net.core.compare.stacked import Stacked
from tile2net.core.compare.vecgrid import VecGridNamespace as VecGridNamespace
from tile2net.core.dir.outdir import Outdir
from tile2net.core.frame.framewrapper import FrameWrapper

if TYPE_CHECKING:
    from tile2net.core.ingrid import InGrid


class Compare(
    FrameWrapper,
):
    left: InGrid
    """InGrid which appears on the left in the comparisons."""
    right: InGrid
    """InGrid which appears on the right in the comparisons."""

    @File
    def file(self):
        """
        Mirrors the `Grid.file` modules, but creates side-by-side comparisons
        of the respective files in the left and right grids.
        """

    @Stacked
    def stacked(self):
        """Just like file, but the comparisons are stacked below the static images."""

    @cached_property
    def name(self) -> str:
        return f'{self.left.name}_vs_{self.right.name}'

    @classmethod
    def from_grids(
            cls,
            left: InGrid,
            right: InGrid,
            outdir: str = None,
            name: str = None
    ) -> Self:
        index = left.index.intersection(right.index)
        frame = pd.DataFrame(index=index)
        out = cls.from_frame(frame)
        if outdir is None:
            outdir = Outdir.from_parent(left.outdir, 'compare')
        out.left = left
        out.right = right
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