from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import *
from typing import Self

import numpy as np
import pandas as pd

from tile2net.core import frame
from tile2net.core.cfg.logger import logger
from tile2net.core.loaders.sample import SampleDataWrapper
from tile2net.core.sampler.benchmark import Benchmark
from tile2net.core.source.remote import Remote
from tile2net.geo.seggrid import vectile, SegGrid
from tile2net.core.util import recursion_block

if TYPE_CHECKING:
    from ..ingrid import InGrid
    from ..vecgrid.vecgrid import VecGrid
    from ..ingrid import InGrid
    from ..ingrid import InGrid
    from . import predict as predict_py
    from tile2net.core.seggrid.minibatch import MiniBatch


class VecTile(
    vectile.VecTile
):

    @frame.column
    def row(self):
        """row within the segtile of this tile"""

        ytile = self.grid.ytile.to_series()
        result = (
            ytile
            .groupby(self.ytile.values)
            .min()
            .loc[self.ytile]
            .rsub(ytile.values)
            .values
        )
        return result

    @frame.column
    def col(self):
        """column within the segtile of this tile"""
        xtile = self.grid.xtile.to_series()
        result = (
            xtile
            .groupby(self.xtile.values)
            .min()
            .loc[self.xtile]
            .rsub(xtile.values)
            .values
        )
        return result

    @property
    def length(self) -> int:
        return self.seggrid.vecgrid.padded.length

    @frame.column
    def grayscale(self) -> pd.Series:
        """seggrid.file broadcasted to grid"""
        result = (
            self.ingrid.seggrid.broadcast.file.pred
            .loc[self.index]
            .values
        )
        return result


def boundary_grid(
        shape: tuple[int, int],
        pad: int = 1,
) -> np.ndarray:
    rows, cols = shape
    r = np.arange(-pad, rows + pad)
    c = np.arange(-pad, cols + pad)
    R, C = np.meshgrid(r, c, indexing='ij')
    coords = np.column_stack((R.ravel(), C.ravel()))
    mask = coords[:, 0] < 0
    mask |= coords[:, 0] >= rows
    mask |= coords[:, 1] < 0
    mask |= coords[:, 1] >= cols
    return coords[mask]


class Broadcast(
    SegGrid
):
    """
    SegGrid extension with broadcasting to handle overlapping vec-tiles.

    While a seg-tile consists of in-tiles, an in-tile may stitch into multiple
    seg-tiles due to padding overlaps. Broadcast overcomes the base SegGrid's 
    one-row-per-tile limitation by allowing multiple entries per in-tile to
    pair them with multiple seg-tiles.

    Handles lazy-loading of broadcast grid with padding:
    >>> Broadcast._get

    See usage:
    >>> InGrid.broadcast
    """
    instance: SegGrid

    def _get(
            self,
            instance: SegGrid,
            owner,
    ) -> Self:
        """
        Lazy-load factory method for accessing Broadcast from SegGrid

        Automatically generates expanded grid with padding to align segmentation
        tiles with vec-tile boundaries. Creates duplicate rows for seg-tiles that
        overlap multiple vec-tiles.

        Returns:
            Broadcast instance with expanded index including padding tiles

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.broadcast
            Broadcast with multiple rows per in-tile where overlaps occur
        """
        if instance is None:
            result = self
            result.instance = instance
            return result
        instance = instance.seggrid
        self.instance = instance
        cache = instance.frame.__dict__
        key = self.__name__

        if key in cache:
            result = cache[key]
        else:
            vecgrid = instance.vecgrid
            seggrid = instance.seggrid.filled
            corners = (
                vecgrid
                .to_corners(seggrid.scale)
                .to_padding()
            )
            vectile = corners.index.repeat(corners.area)
            kwargs = {
                'vectile.xtile': vectile.get_level_values('xtile'),
                'vectile.ytile': vectile.get_level_values('ytile'),
            }

            result = (
                corners
                .to_grid(drop_duplicates=False)
                .frame
                .assign(**kwargs)
                .pipe(Broadcast.from_frame, wrapper=instance)
            )

            instance.frame.__dict__[self.__name__] = result

            d = seggrid.scale - vecgrid.scale
            expected = 2 ** (2 * d) + 4 * 2 ** d + 4
            assert len(result) == expected * len(vecgrid)
            _ = result.vectile.row, result.vectile.col

        result.instance = instance
        return result

    locals().update(__get__=_get)

    @property
    def seggrid(self) -> SegGrid:
        return self.instance.seggrid

    @property
    def vecgrid(self) -> VecGrid:
        return self.instance.vecgrid

    @VecTile
    def vectile(self):
        ...

    @property
    def ingrid(self) -> InGrid:
        return self.instance.ingrid

    @property
    def filled(self):
        return self

    @property
    def sampler(self) -> Benchmark:
        return self.seggrid.sampler

    @property
    def broadcast(self):
        return self

    @recursion_block
    def predict(
            self,
            pred=False,
            prob=False,
            unclipped_prob=False,
            colorized=False,
    ):
        """
        Run semantic segmentation prediction on all tiles in the grid using a subprocess.

        See `predict.py` for the inference subprocess:
            >>> predict_py.main()
        See the forward pass through the network:
            >>> MiniBatch.from_data()

        The subprocess runs predict.py which performs the actual inference.
        Benchmarking is done in the parent process after subprocess completes.

        See the output files:
            >>> grid: InGrid
            >>> grid.seggrid.file.pred
            >>> grid.seggrid.file.prob
            >>> grid.seggrid.file.unclipped_prob

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.predict(output='pred')
            Downloading weights for segmentation...
            Predicting seg-tiles: 100%|██████| 64/64 [02:15<00:00]
            Finished predicting 64 seg-tiles.
        """
        if not self.predict:
            # prevent infinite recursion
            return


        grid = self.ingrid.broadcast
        force = self.cfg.force

        msg = f'Predicting seg-tiles to \n\t{self.ingrid.outdir.seggrid.pred.dir}'
        logger.info(msg)

        if self.cfg.segmentation.stream:
            if not isinstance(grid.source, Remote):
                msg = f"Streaming mode requires a Remote source, got {type(grid.source)}"
                raise TypeError(msg)
            image_paths = grid.source.url
        else:
            image_paths = grid.file.static

        # Instantiate a custom DataFrame which wraps the metadata necessary for prediction
        kwargs = dict(
            image_paths=image_paths,
            index=grid.segtile.index,
            background=0,
            row=grid.segtile.row,
            col=grid.segtile.col,
        )
        with grid.segtile:
            if pred:
                col = grid.segtile.pred
                force |= ~col.map(os.path.exists)
                kwargs['pred_paths'] = col
            if prob:
                col = grid.segtile.prob
                force |= ~col.map(os.path.exists)
                kwargs['prob_paths'] = col
            if unclipped_prob:
                col = grid.segtile.unclipped_prob
                force |= ~col.map(os.path.exists)
                kwargs['unclipped_prob_paths'] = col
            if colorized:
                col = grid.segtile.colorized
                force |= ~col.map(os.path.exists)
                kwargs['colorized_paths'] = col

        kwargs['force'] = force

        wrapper = SampleDataWrapper.from_columns(**kwargs)

        if wrapper.empty:
            msg = f'All seg-tiles are already on disk.'
            logger.info(msg)
            return

        padding = self.cfg.segmentation.pad
        tile_dimension = self.ingrid.dimension

        with (
            tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as cfg_file,
            tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as wrapper_file,
        ):
            cfg_path = cfg_file.name
            wrapper_path = wrapper_file.name

        if not self.cfg.segmentation.stream:
            assert (
                self.ingrid.file.static
                .map(os.path.exists)
                .all()
            )

        # serialize
        self.cfg.to_json(cfg_path)
        wrapper.to_parquet(wrapper_path)

        predict = (
            Path(__file__)
            .resolve()
            .with_name('predict.py')
            .__str__()
        )

        cmd = [
            sys.executable,
            str(predict),
            "--cfg", cfg_path,
            "--wrapper", wrapper_path,
            "--tile-dimension", str(tile_dimension),
            '--padding', str(padding),
        ]

        if self.cfg.segmentation.stream:
            mode_desc = "streaming"
        else:
            mode_desc = "serialized"
        logger.info(f"Launching prediction subprocess ({mode_desc} mode)")
        logger.debug(f"Command: {' '.join(cmd)}")

        with self.benchmark:
            process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=os.environ.copy(),
            )
            process.wait()

        match process.returncode:
            case 0:
                logger.debug(f"Subprocess completed")
            case 130:
                raise KeyboardInterrupt("Prediction interrupted by user")
            case _:
                raise RuntimeError(f"Prediction subprocess failed (Exit Code {process.returncode}).")

        self._write_benchmark_summary()

        # cleanup
        try:
            os.unlink(cfg_path)
            os.unlink(wrapper_path)
            metadata_path = wrapper_path.replace('.parquet', '_metadata.json')
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)
        except Exception as exc:
            logger.warning(f"Could not clean up temp files: {exc}")
