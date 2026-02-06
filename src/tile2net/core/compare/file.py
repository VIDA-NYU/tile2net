from __future__ import annotations

from typing import *

import pandas as pd

from tile2net.core.frame.namespace import namespace
from .. import frame

if TYPE_CHECKING:
    from .grid_namespace import GridNamespace
    from .compare import Compare


class File(namespace):
    instance: GridNamespace
    wrapper: Compare

    @property
    def compare(self) -> Compare:
        return self.wrapper

    @classmethod
    def _compute_comparison(
            cls,
            left_static_file: str,
            right_static_file: str,
            left_colorized_file: str,
            right_colorized_file: str,
            output_file: str,
    ) -> str:
        ...

    @frame.column
    def comparison(self) -> pd.Series:
        compare = self.compare
        left_colorized = compare.left.seggrid.file.colorized
        right_colorized = compare.right.seggrid.file.colorized
        left_static = compare.left.seggrid.file.static
        right_static = compare.right.seggrid.file.static
        FILES = compare.outdir.seggrid.colorized.files(grid=compare)

        if self:
            return FILES
        raise NotImplementedError
