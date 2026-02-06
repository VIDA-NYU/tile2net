from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import *

import pandas as pd
import tifffile
from PIL import Image

from tile2net.core import cfg, frame
from tile2net.core.compare.file import File
from tile2net.logger import logger

if TYPE_CHECKING:
    from .grid_namespace import GridNamespace
    from .compare import Compare


class Stacked(File):
    instance: GridNamespace
    wrapper: Compare

    @property
    def compare(self) -> Compare:
        return self.wrapper

    @classmethod
    def _compute_colorized(
            cls,
            left_static_file: str,
            right_static_file: str,
            left_pred_file: str,
            right_pred_file: str,
            output_file: str,
    ) -> str:
        """
        Create a 2x2 composite grid directly from prediction masks:
        Top: Left Static & Right Static
        Bottom: Left Pred (Colorized on the fly) & Right Pred (Colorized on the fly)
        """
        # Load Static Imagery
        left_static = Image.open(left_static_file).convert('RGB')
        right_static = Image.open(right_static_file).convert('RGB')

        # Load Prediction Masks
        # Reading raw class indices from disk
        left_pred = tifffile.imread(left_pred_file)
        right_pred = tifffile.imread(right_pred_file)

        # Colorize Masks
        # cfg.colormap returns BGR (OpenCV format), so we flip to RGB for PIL
        left_bgr = cfg.colormap(left_pred)
        right_bgr = cfg.colormap(right_pred)

        left_colorized = Image.fromarray(left_bgr[..., ::-1])
        right_colorized = Image.fromarray(right_bgr[..., ::-1])

        # Stitch 2x2 Grid
        width, height = left_static.size
        composited = Image.new('RGB', (width * 2, height * 2))

        # Top Row: Static Imagery
        composited.paste(left_static, (0, 0))
        composited.paste(right_static, (width, 0))

        # Bottom Row: Colorized Predictions
        composited.paste(left_colorized, (0, height))
        composited.paste(right_colorized, (width, height))

        # Downscale to max 1024px width if necessary
        MAX_WIDTH = 1024
        if composited.width > MAX_WIDTH:
            ratio = MAX_WIDTH / composited.width
            new_height = int(composited.height * ratio)
            composited = (
                composited
                .resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
            )

        # Atomic Save
        (
            Path(output_file)
            .parent
            .mkdir(parents=True, exist_ok=True)
        )
        tmp = (
                Path(output_file)
                .parent
                / f'tmp.{Path(output_file).name}'
        )

        try:
            composited.save(str(tmp))
            os.replace(tmp, output_file)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return output_file

    @frame.column
    def colorized(self) -> pd.Series:
        """
        Generates composites by mapping prediction masks to colors on-the-fly.
        """
        compare = self.compare
        left_static = compare.left.seggrid.file.static
        right_static = compare.right.seggrid.file.static

        # Use prediction files (indices) instead of pre-colorized images
        left_pred = compare.left.seggrid.file.pred
        right_pred = compare.right.seggrid.file.pred

        FILES = compare.outdir.seggrid.colorized.files(grid=compare)

        if not self:
            return FILES

        name = (
            str(FILES.name)
            .rsplit('.', 1)[-1]
        )
        path: str = str(Path(FILES.iloc[0]).parent)
        trace = f'{self._trace}.{name}'

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            logger.info(f'{trace} generating {n} composites to\n\t{path}')

            # Filter inputs to only those needing generation
            # Note: We are shadowing the variables here with the filtered subsets
            left_static = left_static.loc[loc]
            right_static = right_static.loc[loc]
            left_pred = left_pred.loc[loc]
            right_pred = right_pred.loc[loc]
            files = FILES.loc[loc]

            max_workers = min(
                len(files),
                (os.cpu_count() or 1) * 2,
                32
            )

            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                it = zip(left_static, right_static, left_pred, right_pred, files)
                futures: dict[Future, str] = {
                    threads.submit(
                        self._compute_colorized,
                        ls, rs, lp, rp, f
                    ): f
                    for ls, rs, lp, rp, f in it
                }
                for future in futures:
                    future.result()

            assert files.map(os.path.exists).all()
        else:
            logger.info(f'{trace} found all {len(loc)} composites in \n\t{path}')

        return FILES
