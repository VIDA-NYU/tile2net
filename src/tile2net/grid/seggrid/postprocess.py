from __future__ import annotations

import os

import pandas as pd

from tile2net.grid import frame
from tile2net.grid.grid import file

if False:
    from tile2net.grid.seggrid import SegGrid


class PostProcess(
    file.File
):
    grid: SegGrid
    """
    Namespace for work-in-progress postprocessing of segmentation results. 
    
    See usage:
        >>> SegGrid.file.postprocess
    """

    @frame.column
    def grayscale(self) -> pd.Series:
        """Segmentation masks, where each pixel is a class id"""
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.postprocess.grayscale.files(grid)
        self.grayscale = files
        if (
            grid.postprocess
            and not files.map(os.path.exists).all()
        ):
            assert (
                grid.filled.index
                .difference(grid.ingrid.broadcast.segtile.index)
                .empty
            )
            grid.file.grayscale = files
            grid.predict()
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def probability(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.postprocess.prob.files(grid)
        self.probability = files
        if (
            grid.postprocess
            and not files.map(os.path.exists).all()
        ):
            assert (
                grid.filled.index
                .difference(grid.ingrid.broadcast.segtile.index)
                .empty
            )
            grid.file.probability = files
            grid.predict()
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def error(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.postprocess.error.files(grid)
        self.error = files
        if (
            grid.postprocess
            and not files.map(os.path.exists).all()
        ):
            assert (
                grid.filled.index
                .difference(grid.ingrid.broadcast.segtile.index)
                .empty
            )
            grid.file.error = files
            grid.predict()
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def colored(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.postprocess.colored.files(grid)
        self.colored = files
        if (
            grid.postprocess
            and not files.map(os.path.exists).all()
        ):
            assert (
                grid.filled.index
                .difference(grid.ingrid.broadcast.segtile.index)
                .empty
            )
            grid.file.colored = files
            grid.predict()
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def intensity(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.postprocess.intensity.files(grid)
        self.intensity = files
        if (
            grid.postprocess
            and not files.map(os.path.exists).all()
        ):
            assert (
                grid.filled.index
                .difference(grid.ingrid.broadcast.segtile.index)
                .empty
            )
            grid.file.intensity = files
            grid.predict()
            assert files.map(os.path.exists).all()
        return files
