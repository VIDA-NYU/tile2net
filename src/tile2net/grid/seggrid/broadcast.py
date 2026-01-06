from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import *

from tile2net.grid.cfg.logger import logger
from tile2net.grid.loaders.sample import SampleDataWrapper
from tile2net.grid.seggrid.minibatch import MiniBatch
from ..util import recursion_block

if False:
    from ..ingrid import InGrid
import pandas as pd

import numpy as np

from . import vectile
from .seggrid import SegGrid
from .. import frame
from ..sampler.benchmark import Benchmark

if False:
    from .seggrid import SegGrid
    from ..vecgrid.vecgrid import VecGrid
    from ..ingrid.ingrid import InGrid


class VecTile(
    vectile.VecTile
):

    @frame.column
    def r(self):
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
    def c(self):
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
        """seggrid.file broadcasted to ingrid"""
        ingrid = self.ingrid
        result = (
            ingrid.seggrid.broadcast.file.pred
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
            self: Broadcast,
            instance: SegGrid,
            owner,
    ) -> Broadcast:
        """
        Lazy-load factory method for accessing Broadcast from SegGrid

        Automatically generates expanded grid with padding to align segmentation
        tiles with vec-tile boundaries. Creates duplicate rows for seg-tiles that
        overlap multiple vec-tiles.

        Returns:
            Broadcast instance with expanded index including padding tiles

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.broadcast
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
            _ = result.vectile.r, result.vectile.c

        result.instance = instance
        return result

    locals().update(
        __get__=_get
    )

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
            probs: Optional[bool] = None
    ):
        """
        Run semantic segmentation prediction on all tiles in the grid using subprocess.

        Args:
            probs: If True, generate probability maps. If False, generate predictions only.
                   If None, raises NotImplementedError.

        This version uses JSON/Parquet serialization instead of pickle, allowing for:
        - No security vulnerabilities from pickle
        - Clean subprocess isolation

        The subprocess runs predict.py which performs the actual inference.
        Benchmarking is done in the parent process after subprocess completes.

        Returns:
            None. See output file paths:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.pred
            >>> ingrid.seggrid.file.prob

        Raises:
            RuntimeError: If subprocess fails or model weights checksum is invalid
            FileNotFoundError: If required model checkpoints are missing
            NotImplementedError: If probs is None

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.predict(probs=False)
            Downloading weights for segmentation...
            Predicting seg-tiles: 100%|██████| 64/64 [02:15<00:00]
            Finished predicting 64 seg-tiles.
        """
        if not self.predict:
            return
        if probs is None:
            raise NotImplementedError("probs parameter must be explicitly set to True or False")
        ingrid = self.ingrid.broadcast
        force = ~ingrid.segtile.pred.map(os.path.exists)
        if probs:
            force |= ~ingrid.segtile.prob.map(os.path.exists)
        force |= self.cfg.force

        wrapper: SampleDataWrapper = SampleDataWrapper.from_tiles(
            infile=ingrid.file.infile,
            mask=[None] * len(ingrid),
            index=ingrid.segtile.index,
            background=0,
            row=ingrid.segtile.r,
            col=ingrid.segtile.c,
            force=force,
        )

        if wrapper.empty:
            msg = f'All seg-tiles are already on disk.'
            logger.info(msg)
            return

        clip = self.ingrid.dimension * self.cfg.segmentation.pad
        with (
            tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as cfg_file,
            tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as wrapper_file,
        ):
            cfg_path = cfg_file.name
            wrapper_path = wrapper_file.name

        assert (
            self.ingrid.file.infile
            .map(os.path.exists)
            .all()
        )

        # serialize
        self.cfg.to_json(cfg_path)
        wrapper.to_parquet(wrapper_path)
        seggrid = wrapper_path.replace('.parquet', '_seggrid.parquet')
        self.broadcast.to_parquet(seggrid)
        predict = (
            Path(__file__)
            .resolve()
            .with_name('predict.py')
            .__str__()
        )

        seggrid = wrapper_path.replace('.parquet', '_seggrid.parquet')
        cmd = [
            sys.executable,
            str(predict),
            "--cfg", cfg_path,
            "--wrapper", wrapper_path,
            "--seggrid", seggrid,
            "--clip", str(clip),
            "--probs", 'true' if probs else 'false'
        ]

        logger.info(f"Launching prediction subprocess: {' '.join(cmd)}")

        with self.benchmark:
            process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=os.environ.copy(),
            )
            process.wait()

        if process.returncode != 0:
            if process.returncode == 130:
                raise KeyboardInterrupt("Prediction interrupted by user")

            raise RuntimeError(
                f"Prediction subprocess failed (Exit Code {process.returncode})."
            )

        self._write_benchmark_summary()

        # cleanup
        try:
            os.unlink(cfg_path)
            os.unlink(wrapper_path)
            metadata_path = wrapper_path.replace('.parquet', '_metadata.json')
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)
            seggrid = wrapper_path.replace('.parquet', '_seggrid.parquet')
            if os.path.exists(seggrid):
                os.unlink(seggrid)
        except Exception as exc:
            logger.warning(f"Could not clean up temp files: {exc}")
