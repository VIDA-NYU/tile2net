from __future__ import annotations

import os

import pandas as pd

from tile2net.grid.source import Local, Remote

if False:
    from tile2net.grid.frame import column
    from tile2net.grid.ingrid import InGrid
    from tile2net.grid.frame import column
    from tile2net.grid.ingrid import InGrid

from .. import frame
from tile2net.basegrid.basegrid import file

if False:
    from .ingrid import InGrid


class File(
    file.File
):
    instance: InGrid
    basegrid: InGrid

    @frame.column
    def static(self):
        grid = self.basegrid
        source = grid.source
        if isinstance(source, Remote):
            files = grid.outdir.ingrid.files(grid)
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

        ingrid = self.basegrid
        files = ingrid.outdir.ingrid.pred.files(ingrid)
        self.pred = files
        if (
                not self
                and (loc := ~files.map(os.path.exists)).any()
        ):
            ingrid = ingrid.loc[loc]
            ingrid._unstitch2file(
                tiles=ingrid.file.pred,
                mosaics=ingrid.segtile.pred,
                row=ingrid.segtile.row,
                col=ingrid.segtile.col,
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
        ingrid = self.basegrid
        files = ingrid.outdir.ingrid.prob.files(ingrid)
        self.prob = files
        loc = ~files.map(os.path.exists)
        if (
                not self
                and loc.any()
        ):
            ingrid = ingrid.loc[loc]
            ingrid._unstitch2file(
                tiles=ingrid.file.prob,
                mosaics=ingrid.segtile.prob,
                row=ingrid.segtile.row,
                col=ingrid.segtile.col,
            )
            loc = ~files.map(os.path.exists)
            msg = f"Files not unstiched: {files[loc]}"
            assert not loc.any(), msg
        return files

    @frame.property
    def network(self):
        file = self.basegrid.outdir.network.parquet
        self.network = file
        if not self:
            _ = self.basegrid.network
        return file

    @frame.property
    def polygons(self):
        file = self.basegrid.outdir.polygons.parquet
        self.polygons = file
        if not self:
            _ = self.basegrid.polygons
        return file
