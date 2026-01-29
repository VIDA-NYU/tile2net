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
from ..basegrid import file
from ...grid import util

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))

if TYPE_CHECKING:
    from .seggrid import SegGrid
    from ..grid import Grid


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


class File(
    file.File
):
    basegrid: SegGrid

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

    @frame.column
    def static(self) -> pd.Series:
        """
        A file for each seg-tile: the stitched input grid.
        Stitches input files when seggrid.file is accessed
        """
        grid = self.basegrid
        files = self.dir.static.files(grid)
        setattr(self, 'static', files)
        if self:
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            self.dir
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Stitching to\n\t{path}'
            logger.info(msg)
            grid = grid.grid
            assert (
                grid.file.static
                .map(os.path.exists)
                .all()
            )
            mosaics = grid.segtile.static
            grid._stitch2file(
                tiles=grid.file.static,
                mosaics=mosaics,
                row=grid.segtile.row,
                col=grid.segtile.col,
            )
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @frame.column
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.file.pred
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.basegrid.broadcast
        files = grid.outdir.seggrid.pred.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.pred = files
        setattr(self, 'pred', files)
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
            .__getattribute__(grid.__name__)
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Predicting to\n\t{path}'
            logger.info(msg)
            grid.predict(probs=False)
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
            >>> grid: Grid
            >>> grid.seggrid.file.prob
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.basegrid.broadcast
        files = grid.outdir.seggrid.prob.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.prob = files
        setattr(self, 'prob', files)
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
            .__getattribute__(grid.__name__)
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Predicting to\n\t{path}'
            logger.info(msg)
            grid.predict(probs=True)
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @frame.column
    def disk_usage(self):
        # todo: include other files
        result = util.path2fsize(self.pred)
        result += util.path2fsize(self.colorized)
        return result
