from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import *

import cv2
import numpy as np
import pandas as pd
import tifffile

from tile2net.grid import frame
from tile2net.grid.basegrid import file

if TYPE_CHECKING:
    from tile2net.grid.seggrid import SegGrid


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
        files: pd.Series = (
            self.basegrid.outdir.seggrid
            .__getattribute__(self.__name__)
            .files(self.basegrid)
        )

        if not files.map(os.path.exists).all():
            def write(prob_path: str, pred_path: str):
                if os.path.exists(pred_path):
                    return

                # Read probability map
                prob = tifffile.imread(prob_path)
                # Compute prediction
                pred = np.argmax(prob, axis=0).astype(np.uint8)

                # Atomic write pattern
                p = Path(pred_path)
                parent = p.parent
                parent.mkdir(parents=True, exist_ok=True)
                tmp = p.parent / f'tmp.{p.name}'
                tmp_str = str(tmp)

                try:
                    success = cv2.imwrite(
                        tmp_str,
                        pred,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1]
                    )
                    if not success:
                        raise RuntimeError(
                            f'imwrite failed for {pred_path} with '
                            f'shape={pred.shape} dtype={pred.dtype}'
                        )
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
