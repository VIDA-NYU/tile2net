from __future__ import annotations

import hashlib
import os
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch

from .postprocess import PostProcess
from .. import frame
from ..grid import file
from ...grid import util

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))

if False:
    from .seggrid import SegGrid


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


class File(
    file.File
):
    grid: SegGrid

    @PostProcess
    def postprocess(self) -> pd.Series:
        """
        Namespace for work-in-progress postprocessing of segmentation results.
        """

    @frame.column
    def static(self) -> pd.Series:
        """
        A file for each seg-tile: the stitched input grid.
        Stitches input files when seggrid.file is accessed
        """
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.static.files(grid)
        self.static = files
        if not files.map(os.path.exists).all():
            ingrid = grid.ingrid
            assert (
                ingrid.file.static
                .map(os.path.exists)
                .all()
            )
            mosaics = ingrid.segtile.static
            ingrid._stitch2file(
                tiles=ingrid.file.static,
                mosaics=mosaics,
                row=ingrid.segtile.row,
                col=ingrid.segtile.col,
            )
            msg = f"Files not stitched: {files[~files.map(os.path.exists)]}"
            assert files.map(os.path.exists).all(), msg

        return files

    @frame.column
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.pred
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid.broadcast
        files = grid.outdir.seggrid.pred.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.pred = files
        self.pred = files
        if (
                not grid.predict
                and not files.map(os.path.exists).all()
        ):
            grid.predict(probs=False)
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.prob
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid.broadcast
        files = grid.outdir.seggrid.prob.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.prob = files
        self.prob = files
        if (
                not bool(grid.predict)
                and not files.map(os.path.exists).all()
        ):
            grid.predict(probs=True)
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def disk_usage(self):
        # todo: include other files
        result = util.path2fsize(self.pred)
        result += util.path2fsize(self.colorized)
        return result

    @staticmethod
    def _load_prob(prob_file: str) -> torch.Tensor:
        """Load probability map from TIFF file. Returns tensor with shape (C, H, W)."""
        arr = tifffile.imread(prob_file).astype(np.float32)
        return torch.from_numpy(arr)

    @staticmethod
    def _load_pred(pred_file: str) -> torch.Tensor:
        """Load prediction mask from PNG file. Returns tensor with shape (H, W)."""
        arr = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(arr)
