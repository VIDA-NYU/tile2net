from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import *

import pandas as pd
import tifffile
from PIL import Image

from tile2net.core import cfg, frame
from tile2net.eval.file import File
from tile2net.logger import logger

if TYPE_CHECKING:
    from .grid_namespace import GridNamespace
    from .eval import Eval


class Stacked(File):
    instance: GridNamespace
    wrapper: Eval

    @property
    def eval(self) -> Eval:
        return self.wrapper

    @classmethod
    def _compute_colorized(
            cls,
            static_files: list[str],
            pred_files: list[str],
            output_file: str,
    ) -> str:
        """
        Create a 2xN composite grid directly from prediction masks:
        Top Row: All Static Images
        Bottom Row: All Predictions (Colorized on the fly)
        """
        n_grids = len(static_files)

        static_images = [
            Image.open(static_file).convert('RGB')
            for static_file in static_files
        ]
        pred_arrays = [
            tifffile.imread(pred_file)
            for pred_file in pred_files
        ]

        # Colorize Masks
        colorized_images = []
        for pred_array in pred_arrays:
            bgr = cfg.colormap(pred_array)
            rgb = Image.fromarray(bgr[..., ::-1])
            colorized_images.append(rgb)

        # stitch 2xN grid
        width, height = static_images[0].size
        composited = Image.new('RGB', (width * n_grids, height * 2))

        # top row: static
        for i, static_img in enumerate(static_images):
            composited.paste(static_img, (i * width, 0))

        # bottom row: colorized
        for i, colorized_img in enumerate(colorized_images):
            composited.paste(colorized_img, (i * width, height))

        # Downscale to max 1024px width if necessary
        MAX_WIDTH = 1024
        if composited.width > MAX_WIDTH:
            ratio = MAX_WIDTH / composited.width
            new_height = int(composited.height * ratio)
            composited = composited.resize(
                (MAX_WIDTH, new_height),
                Image.Resampling.LANCZOS
            )

        parent = Path(output_file).parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp = parent / f'tmp.{Path(output_file).name}'

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
        eval = self.eval

        static_series_list = [
            grid.seggrid.file.static
            for grid in eval.grids
        ]
        pred_series_list = [
            grid.seggrid.file.pred
            for grid in eval.grids
        ]

        FILES = eval.outdir.seggrid.colorized.files(grid=eval)

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

            static_series_list = [
                series.loc[loc]
                for series in static_series_list
            ]
            pred_series_list = [
                series.loc[loc]
                for series in pred_series_list
            ]
            files = FILES.loc[loc]

            max_workers = min(
                len(files),
                (os.cpu_count() or 1) * 2,
                32
            )

            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                it = zip(*static_series_list, *pred_series_list, files)
                futures: dict[Future, str] = {
                    threads.submit(
                        self._compute_colorized,
                        list(row[:len(eval.grids)]),
                        list(row[len(eval.grids):-1]),
                        row[-1]
                    ): row[-1]
                    for row in it
                }
                for future in futures:
                    future.result()

            assert files.map(os.path.exists).all()
        else:
            logger.info(f'{trace} found all {len(loc)} composites in \n\t{path}')

        return FILES
