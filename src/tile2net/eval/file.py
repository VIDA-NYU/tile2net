from __future__ import annotations

from typing import *

import pandas as pd

from tile2net.core.frame.namespace import namespace
from tile2net.core import frame

if TYPE_CHECKING:
    from .grid_namespace import GridNamespace
    from .eval import Eval


class File(namespace):
    instance: GridNamespace
    wrapper: Eval

    @property
    def eval(self) -> Eval:
        return self.wrapper

    @classmethod
    def _compute_comparison(
            cls,
            static_files: list[str],
            colorized_files: list[str],
            output_file: str,
    ) -> str:
        ...

    @frame.column
    def comparison(self) -> pd.Series:
        eval = self.eval
        colorized_series = [
            grid.seggrid.file.colorized
            for grid in eval.grids
        ]
        static_series = [
            grid.seggrid.file.static
            for grid in eval.grids
        ]
        FILES = eval.outdir.seggrid.colorized.files(grid=eval)

        if self:
            return FILES
        raise NotImplementedError
