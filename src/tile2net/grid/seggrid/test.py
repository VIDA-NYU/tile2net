from __future__ import annotations

from typing import *
import os
import logging
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.morphology import reconstruction, dilation, disk
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from tile2net.grid import frame
from tile2net.grid.basegrid import file
from tile2net.grid.seggrid.postprocess import PostProcess

if TYPE_CHECKING:
    from tile2net.grid.seggrid import SegGrid

# The only parameter is the "Reach" of the bridge.
# A radius of 3 means we can bridge a gap of roughly ~6 pixels.
# This is a spatial constraint, not a probability threshold.
BRIDGE_RADIUS = 3


def geodesic_margin_reconstruction(prob_tensor: np.ndarray) -> np.ndarray:
    """
    Applies Morphological Closing by Reconstruction on the Confidence Margin.

    Logic:
    1. Compute Margin = Max(Foreground) - Background.
    2. Lift 'valleys' in the Margin map that are surrounded by high peaks.
    3. Update probabilities based on the new Margin.
    """
    # [K, H, W]
    refined = prob_tensor.astype(np.float32)
    K, H, W = refined.shape

    # Identify Background (Last Channel)
    bg_idx = K - 1
    bg_prob = refined[bg_idx]

    # Identify Max Foreground (Best guess for 'Sidewalk' or other FG class)
    # We take the max across all FG channels to treat them as a unified 'Object'
    fg_prob = np.max(refined[:bg_idx], axis=0)

    # 1. Calculate the Margin Map
    # Range: [-1.0, 1.0]
    # Positive = Foreground wins. Negative = Background wins.
    margin_map = fg_prob - bg_prob

    # 2. Morphological Closing by Reconstruction
    # Ideally, we want to remove "dark spots" (local minima) in the margin map
    # that are surrounded by "bright spots" (high confidence).

    # Step A: Dilate the margin (The "Marker" or "Seed").
    # This defines the maximum potential envelope.
    # It effectively "fills" all gaps with the max value of the neighbors.
    seed = dilation(margin_map, footprint=disk(BRIDGE_RADIUS))

    # Step B: Erode the Seed *into* the original Mask until stability.
    # This is "Reconstruction by Erosion".
    # It preserves the peaks (original high confidence) but prevents the
    # dilated valleys from sinking back down to their original low values,
    # UNLESS the valley is too wide (unconnected to a peak).
    reconstructed_margin = reconstruction(
        seed=seed,
        mask=margin_map,
        method='erosion'
    )

    # 3. Project back to Probability Space
    # We have a new Margin M' = P_fg' - P_bg'
    # We know P_fg' + P_bg' = 1 (approx, ignoring other classes for a moment)
    # Solving for P_fg': P_fg' = (M' + 1) / 2
    # This is a linear projection that respects the new topologically clean margin.

    # We only update pixels where the reconstruction actually changed something
    # (i.e., where we lifted a valley).
    mask_changed = reconstructed_margin > margin_map

    if np.any(mask_changed):
        # Calculate new target foreground probability for changed pixels
        # We clamp it to [0, 1] just in case
        new_fg_prob = (reconstructed_margin[mask_changed] + 1.0) / 2.0

        # We need to distribute this new FG probability among the specific FG classes.
        # We simply scale the existing FG classes to sum to this new total.

        # Get sum of original FG probs at these pixels
        current_fg_sum = np.sum(refined[:bg_idx, mask_changed], axis=0)
        # Avoid div by zero
        current_fg_sum[current_fg_sum == 0] = 1.0

        # Scale factor: New_Total / Old_Total
        scale_factor = new_fg_prob / current_fg_sum

        # Update FG channels
        refined[:bg_idx, mask_changed] *= scale_factor

        # Update BG channel (1 - New_Total)
        refined[bg_idx, mask_changed] = 1.0 - new_fg_prob

    return refined


def _process(args: tuple[str, str]) -> None:
    in_path, out_path = args
    try:
        prob_map = tiff.imread(in_path)
        refined_map = geodesic_margin_reconstruction(prob_map)
        tiff.imwrite(out_path, refined_map.astype(np.float32))
    except Exception as e:
        logging.error(f"Failed to process {in_path}: {e}")


class Test(PostProcess):
    """
    Geodesic Margin Reconstruction:
    Operates on the 'Confidence Margin' (FG - BG) rather than raw probabilities.
    Lifts local minima (bridge collapses) in the margin surface to match
    surrounding peaks using morphological reconstruction.
    """
    instance: file.File
    basegrid: SegGrid

    @frame.column
    def prob(self) -> pd.Series:
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
                    desc=f"{trace} (MARGIN RECON)"
                ))

            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logging.info(msg)

        return files