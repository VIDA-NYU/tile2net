from __future__ import annotations

import os
import os.path

import pandas as pd

from .. import frame, util
from ..basegrid import file

if False:
    from .vecgrid import VecGrid
    from ..grid import Grid


class File(
    file.File
):
    basegrid: VecGrid

    @frame.column
    def static(self) -> pd.Series:
        """
        A file for each seg-tile: the stitched input grid.
        Stitches input files when seggrid.file is accessed
        """
        grid = self.basegrid
        files = grid.outdir.vecgrid.static.files(grid)
        self.static = files
        if (
                not self
                and not files.map(os.path.exists).all()
        ):
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.static.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.static,
                mosaics=seggrid.vectile.static,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
            )
            loc = ~seggrid.vectile.static.map(os.path.exists)
            msg = f"Files not stitched: {files[loc]}"
            assert not loc.any(), msg

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
        files = grid.outdir.vecgrid.pred.files(grid)
        self.pred = files
        if (
                not self
                and not files.map(os.path.exists).all()
        ):
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.pred.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.pred,
                mosaics=seggrid.vectile.pred,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
            )

            loc = ~seggrid.vectile.pred.map(os.path.exists)
            msg = f"Files not stitched: {files[loc]}"
            assert not loc.any(), msg

        return files

    @frame.column
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.
        # TODO: update
        """
        grid = self.basegrid
        files = grid.outdir.vecgrid.prob.files(grid)
        self.prob = files
        if (
                not self
                and not files.map(os.path.exists).all()
        ):
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.prob.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.prob,
                mosaics=seggrid.vectile.prob,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
            )

            loc = ~seggrid.vectile.prob.map(os.path.exists)
            msg = f"Files not stitched: {files[loc]}"
            assert not loc.any(), msg

        return files

    @frame.column
    def network(self) -> pd.Series:
        """
        File-paths to line geometry parquet files extracted from segmentation masks.

        Contains centerline network geometries for each vec-tile, representing
        the skeleton of segmented features like sidewalks and crosswalks. These files are
        created during vectorization and used to build the final network output.

        Returns:
            pd.Series: File paths to line geometry parquet files for each vec-tile

        Example:
            >>> grid: Grid
            >>> grid.vecgrid.file.network
            xtile  ytile
            9915   12120    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        vecgrid = self.basegrid
        files = vecgrid.grid.outdir.vecgrid.network.files(vecgrid)
        self.lines = files
        if (
                not self
                and not vecgrid.vectorize
                and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()
        return files

    @frame.column
    def polygons(self) -> pd.Series:
        """
        File-paths to polygon geometry parquet files extracted from segmentation masks.

        Contains polygon geometries for each vec-tile, representing the
        boundaries of segmented features like sidewalks, crosswalks, and roads. These files
        are created during vectorization and used to build the final polygon output.

        Returns:
            pd.Series: File paths to polygon geometry parquet files for each vec-tile

        Example:
            >>> grid: Grid
            >>> grid.vecgrid.file.polygons
            xtile  ytile
            9915   12120    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        vecgrid = self.basegrid
        files = vecgrid.grid.outdir.vecgrid.polygons.files(vecgrid)
        self.polygons = files
        if (
                not self
                and not vecgrid.vectorize
                and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()

        return files

    @frame.column
    def curbs(self) -> pd.Series:
        # todo: needs documentation
        vecgrid = self.basegrid
        files = vecgrid.grid.outdir.vecgrid.curbs.files(vecgrid)
        self.curbs = files
        if (
                not self
                and not vecgrid.vectorize
                and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()
        return files

    @frame.column
    def disk_usage(self):
        # todo: include other files
        result = util.path2fsize(self.grayscale)
        result += util.path2fsize(self.colorized)
        return result
