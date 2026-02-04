
from __future__ import annotations

from typing import *
from functools import singledispatch

from tile2net.core import frame
from tile2net.core.grid import file
from tile2net.core.seggrid.minibatch import clip_image
from tile2net.core.seggrid.postprocess import PostProcess

if TYPE_CHECKING:
    from tile2net.core.seggrid import SegGrid

import os
import logging
import numpy as np
import pandas as pd
import torch
import tifffile as tiff
from skimage.morphology import reconstruction, disk
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

# Configuration

# High Threshold: Pixels above this are DEFINITELY the class.
# This anchors the object.
HIGH_THRESHOLD = 0.60

# Low Threshold: Pixels above this COULD be the class.
# This allows the object to extend through "weak" areas (the gaps).
LOW_THRESHOLD = 0.10  # Very permissive, because we enforce connectivity.

# The Boost: How much to multiply validated pixels by.
# Since we have validated connectivity, we can be aggressive.
# This effectively says "If it's connected, it IS the object."
CONFIRMED_GAIN = 10.0

# Connectivity: How far can we jump?
# A small radius (1-2) ensures we follow the line tightly.
CONN_FOOTPRINT = disk(2)


@singledispatch
def hysteresis_boost(prob_tensor):
    raise TypeError(f"Unsupported type for hysteresis_boost: {type(prob_tensor)!r}")


@hysteresis_boost.register
def _hysteresis_boost_np(prob_tensor: np.ndarray) -> np.ndarray:
    """
    Applies Hysteresis Thresholding to selectively boost connected low-confidence regions.

    1. Identifies 'Seeds' (High Confidence).
    2. Identifies 'Mask' (Low Confidence candidates).
    3. Reconstructs valid shapes by growing Seeds into Mask.
    4. Boosts ONLY the reconstructed shapes, leaving isolated noise suppressed.
    """
    # [K, H, W]
    refined = prob_tensor.astype(np.float32)
    K, H, W = refined.shape

    # Assume last class is background (-1)
    for k in range(K - 1):
        prob_slice = refined[k]

        # 1. Define the Anchors (High Confidence)
        seeds = prob_slice > HIGH_THRESHOLD

        # 2. Define the Candidates (Low Confidence)
        # We include anything that even remotely looks like the class.
        candidates = prob_slice > LOW_THRESHOLD

        # 3. Connectivity Check (Geodesic Reconstruction)
        # We grow the seeds into the candidates.
        # Any candidate pixel NOT connected to a seed is discarded.
        # This returns a Boolean mask of the "True" object shape.
        valid_mask = reconstruction(seeds, candidates, method='dilation', footprint=CONN_FOOTPRINT)

        # 4. Surgical Boost
        # We apply the gain ONLY to the valid_mask.
        # This lifts the "bridge collapse" (which is in valid_mask)
        # but does NOT lift random background noise (which is not).
        refined[k] = np.where(valid_mask, prob_slice * CONFIRMED_GAIN, prob_slice)

    # R-normalize
    total_prob = np.sum(refined, axis=0, keepdims=True)
    # Safety for pure zero regions (unlikely with background class)
    total_prob[total_prob < 1e-6] = 1.0
    refined /= total_prob

    return refined


@hysteresis_boost.register
def _hysteresis_boost_torch(prob_tensor: torch.Tensor) -> torch.Tensor:
    """Applies Hysteresis Boost to torch tensor."""
    device = prob_tensor.device
    dtype = prob_tensor.dtype

    prob_np = prob_tensor.detach().cpu().numpy()

    if prob_np.ndim == 4:
        batch_size = prob_np.shape[0]
        with ThreadPoolExecutor() as executor:
            refined_list = list(
                executor.map(_hysteresis_boost_np, [prob_np[i] for i in range(batch_size)])
            )
        refined_np = np.stack(refined_list, axis=0)
    else:
        refined_np = _hysteresis_boost_np(prob_np)

    result = torch.from_numpy(refined_np).to(device=device, dtype=dtype)
    return result


def _process(args: tuple[str, str, int]) -> None:
    in_path, out_path, clip = args
    try:
        prob_map = tiff.imread(in_path)
        refined_map = hysteresis_boost(prob_map)
        clipped_map = clip_image(refined_map, clip)
        tiff.imwrite(out_path, clipped_map.astype(np.float32))
    except Exception as e:
        logging.error(f"Failed to process {in_path}: {e}")


class Hysteresis(PostProcess):
    """
    Hysteresis Thresholding: Surgical connectivity enforcement.
    Uses dual-thresholding to validate weak connections without amplifying noise.

    NOTE: AI Generated
    """
    instance: file.File
    grid: SegGrid

    @frame.column
    def prob(self) -> pd.Series:
        print(f'Note: temporary AI Generated code in Hysteresis.prob')
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

            clip = grid.ingrid.dimension * grid.cfg.segmentation.pad
            tasks = [
                (inp, out, clip)
                for inp, out in zip(missing_inputs, missing_outputs)
            ]

            with ProcessPoolExecutor() as executor:
                list(tqdm(
                    executor.map(_process, tasks),
                    total=n,
                    desc=f"{trace} (HYSTERESIS)"
                ))

            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logging.info(msg)

        return files
