from __future__ import annotations

from typing import TYPE_CHECKING

import os

import pandas as pd

from tile2net.grid.seggrid.file import File as SegGridFile
from tile2net.geo.basegrid.file import File as BaseGridFile
from tile2net.grid.cfg.logger import logger
from tile2net.grid import frame

if TYPE_CHECKING:
    from .seggrid import SegGrid


class File(
    SegGridFile,
    BaseGridFile,
):
    instance: SegGrid
    basegrid: SegGrid

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
        assert files.index == grid.index
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
            grid.predict(output='pred')
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

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
    def mask(self) -> pd.Series:
        """
        A file for each seg-tile: the stitched mask from input grid.
        Stitches mask files when seggrid.file.mask is accessed.
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
            grid = grid.grid
            assert (
                grid.file.mask
                .map(os.path.exists)
                .all()
            )
            mosaics = grid.segtile.mask
            grid._stitch2file(
                tiles=grid.file.mask,
                mosaics=mosaics,
                row=grid.segtile.row,
                col=grid.segtile.col,
            )
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \\n\\t{path}'
            logger.info(msg)
        return files
