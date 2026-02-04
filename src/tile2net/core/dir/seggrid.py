from __future__ import annotations

from typing import *

from .grid import Grid

if TYPE_CHECKING:
    from ..seggrid import seggrid

class SegGrid(
    Grid,
):
    @property
    def summary(self) -> str:
        """
        See:
            >>> seggrid.SegGrid._write_benchmark_summary
        """
        result = f'{self.dir}/summary.txt'
        return result
