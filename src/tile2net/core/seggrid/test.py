from __future__ import annotations

from typing import *

from tile2net.core import frame
from tile2net.core.grid import file
from tile2net.core.seggrid.postprocess import PostProcess

import os
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


if TYPE_CHECKING:
    from tile2net.core.seggrid import SegGrid


def _process(args: tuple[str, str]) -> None:
    ...


class Test(PostProcess):
    """
    NOTE: AI Generated
    """
    instance: file.File
    grid: SegGrid

    @frame.column
    def prob(self) -> pd.Series:
        print(f'Note: temporary AI Generated code in test.prob')
        grid = self.grid
        inputs: pd.Series = grid.file.unclipped_prob
        dir = self.dir.prob
        files = dir.files(grid)
        setattr(self, 'prob', files)

        if self:
            return files

        name = str(files.name).rsplit('.', 1)[-1]
        path: str = dir.dir
        trace = f'{self._trace}.{name}'
        loc = ~files.map(os.path.exists)

        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Postprocessing to\n\t{path}'
            logging.info(msg)

            os.makedirs(path, exist_ok=True)

            missing_inputs = inputs[loc]
            missing_outputs = files[loc]

            tasks = list(zip(missing_inputs, missing_outputs))

            with ProcessPoolExecutor() as executor:
                list(tqdm(
                    executor.map(_process, tasks),
                    total=n,
                    desc=f"{trace}"
                ))

            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logging.info(msg)

        return files

################################################################################