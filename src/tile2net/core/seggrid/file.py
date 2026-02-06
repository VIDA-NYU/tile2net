from __future__ import annotations

import hashlib
import os
import sys
from typing import *

import pandas as pd

from tile2net.logger import logger
from .test import Test
from .gac import GAC
from .gmb import GMB
from .. import frame
from ..grid import file
from tile2net.core import util
from .hysteresis import Hysteresis

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))

if TYPE_CHECKING:
    from .seggrid import SegGrid
    from ..ingrid import InGrid


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

    @Test
    def test(self) -> pd.Series:
        """
        Namespace for work-in-progress postprocessing of segmentation results.
        """

    @GAC
    def gac(self) -> pd.Series:
        """Grayscale Area Closing postprocessing namespace."""

    @GMB
    def gmb(self) -> pd.Series:
        """Graph-based Minimum Spanning Tree postprocessing namespace."""

    @Hysteresis
    def hysteresis(self) -> pd.Series:
        """Hysteresis Thresholding postprocessing namespace."""

    @property
    def static(self) -> pd.Series:
        """
        A file for each seg-tile: the stitched input grid.
        Stitches input files when seggrid.file is accessed
        """
        return self.grid.ingrid.file.static

    @frame.column
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.file.pred
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid.broadcast
        files = grid.outdir.seggrid.pred.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.pred = files
        if (
                self
                or bool(grid.predict)
        ):
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            grid.outdir
            .__getattribute__(grid._name)
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Predicting to\n\t{path}'
            logger.info(msg)
            setattr(self, 'pred', files)
            grid.predict()
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @frame.column
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.file.prob
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid.broadcast
        files = grid.outdir.seggrid.prob.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.prob = files
        if (
                self
                or bool(grid.predict)
        ):
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            grid.outdir
            .__getattribute__(grid._name)
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Predicting to\n\t{path}'
            logger.info(msg)
            setattr(self, 'prob', files)
            grid.predict(prob=True)
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @frame.column
    def unclipped_prob(self) -> pd.Series:
        """
        File-paths to unclipped probability maps for postprocessing.

        Example:
            >>> grid: InGrid
            >>> grid.seggrid.file.unclipped_prob
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid.broadcast
        files = grid.outdir.seggrid.unclipped_prob.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.unclipped_prob = files
        if (
                self
                or bool(grid.predict)
        ):
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            grid.outdir
            .__getattribute__(grid._name)
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Predicting to\n\t{path}'
            logger.info(msg)
            setattr(self, 'unclipped_prob', files)
            grid.predict(unclipped_prob=True)
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @property
    def mask(self) -> pd.Series:
        """
        A file for each seg-tile: the stitched mask from input grid.
        Stitches mask files when seggrid.file.mask is accessed.
        """
        return self.grid.ingrid.file.mask

    @frame.column
    def disk_usage(self):
        # todo: include other files
        result = util.path2fsize(self.pred)
        result += util.path2fsize(self.colorized)
        return result
