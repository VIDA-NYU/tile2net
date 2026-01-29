from __future__ import annotations

from typing import *

from tile2net.grid import frame
from tile2net.grid.basegrid import file
from tile2net.grid.seggrid.postprocess import PostProcess
from tile2net.grid.seggrid.minibatch import clip_image

if TYPE_CHECKING:
    from tile2net.grid.seggrid import SegGrid

import os
import logging
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.morphology import reconstruction, dilation, disk, square
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# We use a small footprint for the potential mask to bridge small local disconnects
# before the geodesic propagation takes over.
# A small connectivity prevents merging unrelated parallel sidewalks.
CONNECTIVITY_FOOTPRINT = disk(3)

# Multiplier to force foreground classes to dominate background.
# This is the "Amplification" you identified as critical.
FOREGROUND_GAIN = 6.0


def geodesic_masked_boosting(prob_tensor: np.ndarray) -> np.ndarray:
    """
    Applies Morphological Reconstruction by Dilation followed by Foreground Gain.
    This allows high-confidence areas to 'flow' into low-confidence gaps
    without the fixed geometric constraints of a standard closing.
    """
    # [K, H, W]
    refined = prob_tensor.astype(np.float32)
    K, H, W = refined.shape

    # Assume last class is background (-1)
    for k in range(K - 1):
        channel = refined[k]

        # 1. Create the 'Mask' (The Potential Surface).
        # We dilate the current channel. This bridges the gaps in the 'potential' space.
        # If pixel A is 0.9 and B is 0.9, and the gap between them is 0.4,
        # The dilation lifts the gap to 0.9 (depending on footprint size).
        # We use a modest footprint to define "potential connectivity".
        mask = dilation(channel, footprint=CONNECTIVITY_FOOTPRINT)

        # 2. Perform Reconstruction.
        # We use the original channel as the 'Seed'.
        # Ideally, we want the seed to grow into the mask.
        # However, standard reconstruction (dilation method) limits the seed to the mask.
        # Since our Mask > Seed (due to dilation), the Seed values will
        # propagate outward into the gaps defined by the Mask.
        reconstructed = reconstruction(seed=channel, mask=mask, method='dilation')

        # 3. Apply the Gain.
        # Now that the gaps have been lifted to the level of their neighbors
        # (topologically restored), we amplify them to beat the background.
        refined[k] = reconstructed * FOREGROUND_GAIN

    # R-normalize
    # Recalculate denominator so probabilities sum to 1.0
    total_prob = np.sum(refined, axis=0, keepdims=True)
    # Avoid division by zero
    total_prob[total_prob < 1e-6] = 1.0
    refined /= total_prob

    return refined


def _process(args: tuple[str, str, int]) -> None:
    in_path, out_path, clip = args
    try:
        prob_map = tiff.imread(in_path)
        refined_map = geodesic_masked_boosting(prob_map)
        clipped_map = clip_image(refined_map, clip)
        tiff.imwrite(out_path, clipped_map.astype(np.float32))
    except Exception as e:
        logging.error(f"Failed to process {in_path}: {e}")


class GMB(PostProcess):
    """
    Geodesic Masked Boosting: Topological reconstruction with aggressive probability gain.
    NOTE: AI Generated
    """
    instance: file.File
    basegrid: SegGrid

    @frame.column
    def prob(self) -> pd.Series:
        print(f'Note: temporary AI Generated code in GMB.prob')
        grid = self.basegrid
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
            msg = f'{trace} found {n} missing files. Predicting to\n\t{path}'
            logging.info(msg)

            os.makedirs(path, exist_ok=True)

            missing_inputs = inputs[loc]
            missing_outputs = files[loc]

            clip = grid.grid.dimension * grid.cfg.segmentation.pad
            tasks = [
                (inp, out, clip)
                for inp, out in zip(missing_inputs, missing_outputs)
            ]

            with ProcessPoolExecutor() as executor:
                list(tqdm(
                    executor.map(_process, tasks),
                    total=n,
                    desc=f"{trace} (GEODESIC)"
                ))

            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logging.info(msg)

        return files
