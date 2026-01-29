from __future__ import annotations

from typing import *

from tile2net.grid import frame
from tile2net.grid.basegrid import file
from tile2net.grid.seggrid.postprocess import PostProcess

if TYPE_CHECKING:
    from tile2net.grid.seggrid import SegGrid

import os
import logging
import numpy as np
import pandas as pd
import tifffile as tiff
import scipy.ndimage as ndi
from skimage.morphology import disk
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Radius of 15 means gaps up to ~30 pixels wide will be bridged.
CLOSING_RADIUS = 2
# Multiplier to force foreground classes to dominate background.
FOREGROUND_GAIN = 6.


def grayscale_area_closing(prob_tensor: np.ndarray) -> np.ndarray:
    """ Applies Grayscale Closing and Foreground Gain. """
    # [K, H, W]
    refined = prob_tensor.astype(np.float32)
    K, H, W = refined.shape

    # Create the structural element
    footprint = disk(CLOSING_RADIUS)

    # Assume last class is background (-1)
    for k in range(K - 1):
        refined[k] = (
            # fills holes in the probabilty surface; if pixels A and B are high prob,
            # and the space between them is low prob, this sets the space to max(A, B)
            ndi.grey_closing(refined[k], structure=footprint)
            # forcefully increase confidence to beat the background class
            .mul(FOREGROUND_GAIN)
        )

    # R-normalize
    # We modified the numerators, so we must re-calculate the denominator (sum)
    # so that probabilities sum to 1.0 again.
    total_prob = np.sum(refined, axis=0, keepdims=True)
    total_prob[total_prob == 0] = 1.0
    refined /= total_prob

    return refined


def _process(args: tuple[str, str]) -> None:
    in_path, out_path = args
    try:
        prob_map = tiff.imread(in_path)
        refined_map = grayscale_area_closing(prob_map)
        tiff.imwrite(out_path, refined_map.astype(np.float32))
    except Exception as e:
        logging.error(f"Failed to process {in_path}: {e}")


class GAC(PostProcess):
    """NOTE: AI Generated"""
    instance: file.File
    basegrid: SegGrid

    @frame.column
    def prob(self) -> pd.Series:
        print('Note: temporary AI Generated code in GAC.prob')
        grid = self.basegrid
        inputs: pd.Series = grid.file.prob
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
            msg = f'{trace} found {n} missing files. Predicting to\n\t{path}'
            logging.info(msg)

            os.makedirs(path, exist_ok=True)

            missing_inputs = inputs[loc]
            missing_outputs = files[loc]

            tasks = list(zip(missing_inputs, missing_outputs))

            with ProcessPoolExecutor() as executor:
                list(tqdm(
                    executor.map(_process, tasks),
                    total=n,
                    desc=f"{trace} (NUCLEAR)"
                ))

            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logging.info(msg)

        return files
