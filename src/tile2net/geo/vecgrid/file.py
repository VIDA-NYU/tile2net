from __future__ import annotations

from typing import TYPE_CHECKING

import os

import pandas as pd

import tile2net.geo.basegrid.file
import tile2net.grid.vecgrid.file
from tile2net.grid.cfg.logger import logger
from tile2net.grid import frame

if TYPE_CHECKING:
    from .vecgrid import VecGrid


class File(
    tile2net.grid.vecgrid.file.File,
    tile2net.geo.basegrid.file.File,
):
    instance: VecGrid
    basegrid: VecGrid

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
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.static.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.static,
                mosaics=seggrid.vectile.static,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
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
        # TODO: update

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        """
        grid = self.basegrid
        files = self.dir.pred.files(grid)
        setattr(self, 'pred', files)
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
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.pred.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.pred,
                mosaics=seggrid.vectile.pred,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
            )
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @frame.column
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.
        # TODO: update
        """
        grid = self.basegrid
        files = self.dir.prob.files(grid)
        setattr(self, 'prob', files)
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
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.prob.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.prob,
                mosaics=seggrid.vectile.prob,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
            )
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @frame.column
    def mask(self) -> pd.Series:
        """
        A file for each vec-tile: the stitched mask from seg-tiles.
        Stitches mask files when vecgrid.file.mask is accessed.
        """
        grid = self.basegrid
        files = self.dir.mask.files(grid)
        setattr(self, 'mask', files)
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
            msg = f'{trace} found {n} missing files. Stitching to\\n\\t{path}'
            logger.info(msg)
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.mask.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.mask,
                mosaics=seggrid.vectile.mask,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
            )
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \\n\\t{path}'
            logger.info(msg)
        return files
