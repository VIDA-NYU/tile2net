from __future__ import annotations
from pathlib import Path
import contextlib

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
        """Static imagery from the source."""
        grid = self.basegrid
        source = grid.source
        if isinstance(source, Remote):
            files = grid.outdir.source.static.files(grid)
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

    @contextlib.contextmanager
    def _static_peek(self):
        key = self.basegrid.__class__.file.static.key
        grid = self.basegrid
        delete = key not in grid.columns
        with self:
            yield
        if delete:
            del grid[key]

    @frame.property
    def sample(self) -> str:
        """A sample filepath from static; a singular saved file instead of a whole Series"""
        grid = self.basegrid
        source = grid.source
        if not isinstance(source, Remote):
            raise TypeError(f"Unsupported source type: {type(source)}")
        files = source.download_one()
        try:
            sample = next(
                p
                for p in files
                if Path(p).is_file()
            )
        except StopIteration:
            raise FileNotFoundError('No image files found to infer dimension.')
        return sample

    @frame.column
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        # TODO: update
        """

        grid = self.basegrid
        files = grid.outdir.project.pred.files(grid)
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
        files = grid.outdir.project.prob.files(grid)
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
        file = self.basegrid.outdir.project.network.parquet
        self.network = file
        if not self:
            _ = self.basegrid.network
        return file

    @frame.property
    def polygons(self):
        file = self.basegrid.outdir.project.polygons.parquet
        self.polygons = file
        if not self:
            _ = self.basegrid.polygons
        return file
