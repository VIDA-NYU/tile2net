from __future__ import annotations

import copy
import os
from functools import *
from pathlib import Path
from typing import *
from typing import TYPE_CHECKING

from tile2net.geo.basegrid import basegrid
from tile2net.geo.seggrid import delayed
from tile2net.geo.seggrid.padded import Padded
from tile2net.geo.seggrid.vectile import VecTile
from tile2net.core.cfg.logger import logger
from tile2net.core.frame.namespace import namespace
from tile2net.core.sampler.benchmark import Benchmark
from tile2net.core.seggrid import seggrid
from tile2net.core.seggrid.file import File

if TYPE_CHECKING:
    from .filled import Filled
    from .broadcast import Broadcast
    from ..grid import Grid
    from . import predict as predict_py


class SegGrid(
    seggrid.SegGrid,
    basegrid.BaseGrid,
):
    """
    "Segmentation Grid" (SegGrid), comprised of "Segmentation Tiles" (seg-tiles).
    Each seg-tile is a larger tile composed of multiple Grid tiles, used for
    semantic segmentation prediction of street infrastructure.

    SegGrid tiles are typically larger than Grid tiles (e.g., 1024x1024 pixels 
    vs 256x256 pixels) to provide sufficient context for accurate neural network 
    predictions. Each seg-tile covers an area equivalent to multiple in-tiles.

    Example:
        >>> grid: Grid
        >>> grid.seggrid
        SegGrid:
                       lonmin        latmax        lonmax        latmin
        xtile ytile
        79320 96960 -7.911538e+06  5.214840e+06 -7.911385e+06  5.214687e+06
        [64 rows x 8 columns]

    SegGrid handles:
    - Grouping in-tiles into larger seg-tiles (stitching)
    - Running model inference for semantic segmentation
    - Padding tiles to provide more context during prediction

    Handles lazy-loading of SegGrid from Grid:
        >>> SegGrid._get

    See usage:
        >>> Grid.seggrid
    """
    __name__ = 'seggrid'
    instance: Grid

    def _get(
            self,
            instance: Grid,
            owner: type[Grid],
    ) -> SegGrid:
        """
        Lazy-load factory method for accessing SegGrid from Grid.

        Automatically initializes SegGrid using configuration parameters if not already set.
        Uses cached value if available, otherwise calls Grid.set_segmentation() with
        parameters from cfg.segmentation (scale, length, or dimension).

        Returns:
            SegGrid instance configured for segmentation operations

        Example:
            >>> grid: Grid
            >>> grid.seggrid
            SegGrid:
                           lonmin        latmax        lonmax        latmin
            xtile ytile
            79320 96960 -7.911538e+06  5.214840e+06 -7.911385e+06  5.214687e+06
        """
        self = namespace._get(self, instance, owner)
        if instance is None:
            return copy.copy(self)
        if not instance.source:
            msg = f'Cannot generate {self.__name__} in sourceless mode.'
            raise ValueError(msg)
        cache = instance.frame.__dict__
        key = self.__name__
        if key in cache:
            result = cache[key]

        else:
            msg = (
                f'grid.{self.__name__} has not been set. You may '
                f'customize the segmentation functionality by using '
                f'`Ingrid.set_segmentation`'
            )
            logger.info(msg)

            instance = instance.set_segmentation()
            result = instance.seggrid

        result.instance = instance

        return result

    locals().update(__get__=_get)

    @cached_property
    def length(self) -> int:
        """
        Number of Grid tiles that comprise one dimension of a segmentation tile

        For example, if Grid uses zoom 20 and SegGrid uses zoom 18, each SegGrid
        tile is 2^2 = 4 Grid tiles wide.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.length
            4
        """
        grid = self.basegrid.grid
        result = 2 ** (grid.scale - self.scale)
        return result

    @cached_property
    def dimension(self):
        """
        Pixel dimension of each segmentation tile

        For example, if Grid tile are 256x256 pixels and length is 4,
        segmentation tiles are 1024x1024 pixels.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.dimension
            1024
        """
        seggrid = self.basegrid
        grid = seggrid.grid
        result = grid.dimension * self.length
        return result

    @property
    def basegrid(self):
        """Reference to the parent instance."""
        return self.instance

    @property
    def grid(self) -> Grid:
        """Reference to the parent Grid instance."""
        return self.instance

    @property
    def cfg(self):
        """Reference to the configuration object."""
        return self.grid.cfg

    @property
    def static(self):
        """Reference to static assets."""
        return self.grid.static

    @VecTile
    def vectile(self):
        """
        Namespace for vec-tiles aligned with seg-tiles.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.vectile.xtile
            xtile  ytile
            79320  96960    9915

            >>> grid.seggrid.vectile.ytile
            xtile  ytile
            79320  96960    12120

            >>> grid.seggrid.vectile.dimension
            8192
        """

    @File
    def file(self):
        """
        Namespace container for files aligned with the tiles of a Grid.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.file.Colorized
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """

    @property
    def seggrid(self) -> Self:
        """Reference to self as SegGrid."""
        return self

    @property
    def skip(self):
        """Boolean mask indicating which tiles should be skipped during inference."""
        result = ~self.file.pred.apply(os.path.exists)
        return result

    @delayed.Filled
    def filled(self) -> Filled:
        """
        Returns SegGrid extended with additional tiles needed for alignment with VecGrid.
        The filled grid ensures complete coverage and no missing data for all vec-tiles.

        When the vectorization grid (VecGrid) requires a different tiling scheme than
        the segmentation grid, this property automatically fills in any missing seg-tiles
        to ensure complete coverage. This prevents gaps in the segmentation output that
        would otherwise cause issues during vectorization.

        Example:
            >>> grid: Grid
            >>> grid.seggrid
            64
            >>> grid.seggrid.filled
            72

        See also:
            >>> Filled._get
        """

    @delayed.Broadcast
    def broadcast(self) -> Broadcast:
        """
        Handles one-to-many relationships between in-tiles and seg-tiles due to overlaps.

        While the base SegGrid dataframe has one row per unique seg-tile, an individual
        in-tile may belong to multiple overlapping seg-tiles (especially when padding is
        used). The broadcast extension creates a view where each in-tile-to-seg-tile
        membership gets its own row, enabling proper alignment for batch processing.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.broadcast
                           segtile.xtile  segtile.ytile  segtile.r  segtile.c
            xtile  ytile
            317275 387839          79319          96959          4          0
                   387839          79319          96960          0          0

        See also:
           >>> Broadcast._get
        """

    @Padded
    def padded(self):
        """
        Namespace for padded segmentation operations

        Handles padding around seg-tiles to avoid edge artifacts during
        neural network inference. Padding is configurable via cfg.segmentation.pad.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.padded.Static
            >>> grid.seggrid.padded.length
        """

    def _write_benchmark_summary(self) -> None:
        """Write segmentation benchmark summary to file and save detailed CSV."""
        try:
            seg_s = self.benchmark.samples
            seg_vals = {
                'elapsed': seg_s.time_elapsed,
                'gpu_avg': seg_s.avg_gpu,
                'gpu_max': seg_s.max_gpu,
                'vram_avg': seg_s.avg_vram,
                'vram_max': seg_s.max_vram,
                'ram_avg': seg_s.avg_ram,
                'ram_max': seg_s.max_ram,
                'cpu_avg': seg_s.avg_cpu,
                'cpu_max': seg_s.max_cpu,
            }

            def _fmt_pct(v: float | None) -> str:
                return "—" if v is None else f"{v:.1f}%"

            def _fmt_duration(v: float | None) -> str:
                if v is None:
                    return "—"
                secs = float(v)
                if secs < 0:
                    secs = 0.0
                ms = secs * 1000.0
                if secs < 1e-3:
                    return "0s"
                if secs < 1.0:
                    return f"{ms:.0f}ms"
                days = int(secs // 86400)
                secs -= days * 86400
                hours = int(secs // 3600)
                secs -= hours * 3600
                minutes = int(secs // 60)
                secs -= minutes * 60
                parts = []
                if days:
                    parts.append(f"{days}d")
                if hours and len(parts) < 2:
                    parts.append(f"{hours}h")
                if minutes and len(parts) < 2:
                    parts.append(f"{minutes}m")
                if len(parts) < 2:
                    parts.append(f"{secs:.1f}s" if secs < 10 else f"{int(secs)}s")
                return " ".join(parts)

            lines = []
            lines.append("Segmentation benchmark")
            lines.append("======================")
            if seg_vals['elapsed'] is not None:
                lines.append(f"Time Elapsed: {_fmt_duration(seg_vals['elapsed'])}")
            lines.append(f"GPU Compute: avg {_fmt_pct(seg_vals['gpu_avg'])}, max {_fmt_pct(seg_vals['gpu_max'])}")
            lines.append(f"VRAM Usage: avg {_fmt_pct(seg_vals['vram_avg'])}, max {_fmt_pct(seg_vals['vram_max'])}")
            lines.append(f"RAM Usage:  avg {_fmt_pct(seg_vals['ram_avg'])}, max {_fmt_pct(seg_vals['ram_max'])}")
            lines.append(f"CPU Usage:  avg {_fmt_pct(seg_vals['cpu_avg'])}, max {_fmt_pct(seg_vals['cpu_max'])}")

            summary_path = Path(self.outdir.seggrid.summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            # save detailed CSV with all samples
            # todo: have summary.txt and benchmark.csv controlled by cfg
            # csv_path = summary_path.parent / "benchmark.csv"
            # self.benchmark.save_csv(str(csv_path))
            # logger.info(f"Saved detailed benchmark data to {csv_path}")

        except Exception as exc:
            logger.warning(f"Could not write segmentation benchmark summary: {exc}")

    @cached_property
    def benchmark(self):
        """
        Benchmark for segmentation operations (GPU, VRAM, RAM, CPU).
        """
        result = Benchmark(include_gpu=True)
        return result

    @cached_property
    def disk_usage(self) -> int:
        """Total disk space used by all segmentation files in bytes."""
        result = self.broadcast.file.disk_usage.sum()
        return result

    @cached_property
    def time_usage(self) -> float:
        """Time spent on segmentation operations in seconds."""
        return 0.

    def predict(
            self,
            output: Literal['unclipped_prob', 'prob', 'pred'] = 'pred',
    ) -> None:
        """
        Run semantic segmentation prediction on all tiles in the grid using subprocess.

        Args:
            output:
                'unclipped_prob':
                    Serialize unclipped probabilities, clipped probabilities, and predictions.
                'prob':
                    Serialize clipped probabilities and predictions.
                'pred':
                    Serialize predictions only.

        See `predict.py` for the inference subprocess:
            >>> predict_py.main()

        The subprocess runs predict.py which performs the actual inference.
        Benchmarking is done in the parent process after subprocess completes.

        See the output files:
            >>> grid: Grid
            >>> grid.seggrid.file.pred
            >>> grid.seggrid.file.prob
            >>> grid.seggrid.file.unclipped_prob

        Example:
            >>> grid: Grid
            >>> grid.seggrid.predict(output='pred')
            Downloading weights for segmentation...
            Predicting seg-tiles: 100%|██████| 64/64 [02:15<00:00]
            Finished predicting 64 seg-tiles.
        """
        return self.broadcast.predict(output=output)
