from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import tifffile

from tile2net.grid import frame
from tile2net.grid.basegrid import file
from tile2net.grid.dir.dir import Dir
from tile2net.logger import logger

if TYPE_CHECKING:
    from tile2net.grid.seggrid import SegGrid
    from tile2net.grid.dir.basegrid import PostProcessedOutputs


class PostProcess(
    file.File
):
    instance: file.File
    basegrid: SegGrid
    """
    Namespace for work-in-progress postprocessing of segmentation results. 
    """

    @property
    def basegrid(self) -> SegGrid:
        return self.instance.basegrid

    @frame.column
    def pred(self) -> pd.Series:
        probs: pd.Series = self.prob
        files: pd.Series = self.dir.pred.files(self.basegrid)

        if not files.map(os.path.exists).all():
            def write(prob_path: str, pred_path: str):
                if os.path.exists(pred_path):
                    return

                prob = tifffile.imread(prob_path)
                pred = (
                    prob
                    .argmax(axis=0)
                    .astype(np.uint8)
                )

                # Atomic write
                p = Path(pred_path)
                parent = p.parent
                parent.mkdir(parents=True, exist_ok=True)
                tmp = p.parent / f'tmp.{p.name}'
                tmp_str = str(tmp)

                try:
                    tifffile.imwrite(tmp_str, pred, compression='zlib')
                    os.replace(tmp, p)
                except Exception:
                    if tmp.exists():
                        tmp.unlink()
                    raise

            todo = {
                prob_path: pred_path
                for prob_path, pred_path in zip(probs, files)
                if not os.path.exists(pred_path)
            }

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(write, prob_p, pred_p)
                    for prob_p, pred_p in todo.items()
                ]
                for future in futures:
                    future.result()

            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def colorized_sidebyside(self) -> pd.Series:
        """
        Original colorized segmentation on the left and post-processed on the right.
        """
        grid = self.basegrid
        dir: Dir = self.dir.colorized_sidebyside

        FILES = dir.files(grid)
        setattr(self, 'colorized_sidebyside', FILES)
        if self:
            return FILES

        name = (
            str(FILES.name)
            .rsplit('.', 1)[-1]
        )
        path: str = dir.dir
        trace = f'{self._trace}.{name}'

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Generating to\n\t{path}'
            logger.info(msg)

            before = grid.file.colorized.loc[loc]

            after = (
                grid.file
                .__getattribute__(self.__name__)
                .colorized
                .loc[loc]
            )
            assert after.name == f'file.{self.__name__}.colorized'

            type(grid.file.__getattribute__(self.__name__))
            files = FILES.loc[loc]

            max_workers = min(len(files), grid.cfg.compress_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                it = zip(before, after, files)
                futures: dict[Future, str] = {
                    threads.submit(self._compute_sidebyside, b, a, f): f
                    for b, a, f in it
                }
                for future in futures:
                    future.result()

            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return FILES

    @property
    def dir(self) -> PostProcessedOutputs:
        return (
            super().dir
            .__getattribute__(self.__name__)
        )
