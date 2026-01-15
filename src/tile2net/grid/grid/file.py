from __future__ import annotations

import os

import pandas as pd

from tile2net.grid.source import Local, Remote

if False:
    from tile2net.grid.frame import column
    from tile2net.grid.grid import Grid
    from tile2net.grid.frame import column
    from tile2net.grid.grid import Grid

from .. import frame
from tile2net.grid.basegrid import file

if False:
    from .grid import Grid


class File(
    file.File
):
    instance: Grid
    basegrid: Grid

    @frame.column
    def static(self):
        grid = self.basegrid
        source = grid.source
        if isinstance(source, Remote):
            files = grid.outdir.static.files(grid)
        elif isinstance(source, Local):
            files = source.files(grid)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
        grid.file.static = files

        if (
                isinstance(source, Remote)
                and not self
                and not source.download
                and not files.map(os.path.exists).all()
        ):
            source.download()
        return files

    @frame.column
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        # TODO: update
        """

        grid = self.basegrid
        files = grid.outdir.namedir.pred.files(grid)
        self.pred = files
        loc = ~files.map(os.path.exists)
        if (
                not self
                and loc.any()
        ):
            grid = grid.loc[loc]
            grid._unstitch2file(
                tiles=grid.file.pred,
                mosaics=grid.segtile.pred,
                row=grid.segtile.row,
                col=grid.segtile.col,
            )
            loc = ~files.map(os.path.exists)
            msg = f"Files not unstiched: {files[loc]}"
            assert not loc.any(), msg
        return files

    @frame.column
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.
        # TODO: update
        """
        grid = self.basegrid
        files = grid.outdir.namedir.prob.files(grid)
        loc = ~files.map(os.path.exists)
        if (
                not self
                and loc.any()
        ):
            grid = grid.loc[loc]
            grid._unstitch2file(
                tiles=grid.file.prob,
                mosaics=grid.segtile.prob,
                row=grid.segtile.row,
                col=grid.segtile.col,
            )
            loc = ~files.map(os.path.exists)
            msg = f"Files not unstiched: {files[loc]}"
            assert not loc.any(), msg
        return files

    @frame.property
    def network(self):
        file = self.basegrid.outdir.namedir.network.parquet
        self.network = file
        if not self:
            _ = self.basegrid.network
        return file

    @frame.property
    def polygons(self):
        file = self.basegrid.outdir.namedir.polygons.parquet
        self.polygons = file
        if not self:
            _ = self.basegrid.polygons
        return file
