from __future__ import annotations

import os
import os.path

import pandas as pd

from .. import frame
from ..grid import file

# from ...grid.util import recursion_block

if False:
    from .vecgrid import VecGrid
    from ..ingrid import InGrid


class File(
    file.File
):
    grid: VecGrid

    @frame.column
    def grayscale(self) -> pd.Series:
        """
        File-paths to stitched grayscale segmentation masks for each vec-tile.

        Returns:
            pd.Series: File paths to grayscale segmentation masks for each vec-tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.file.pred
            xtile  ytile
            9915   12120    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.pred.files(vecgrid)
        self.grayscale = files
        if not files.map(os.path.exists).all():
            seggrid = vecgrid.seggrid.broadcast
            outgrid = vecgrid

            # preemptively predict so logging appears more sequential
            # else you get "now stitching" before "now predicting"
            _ = seggrid.file.pred

            vecgrid._stitch_to_file(
                small_grid=seggrid,
                big_grid=outgrid,
                r=seggrid.vectile.r,
                c=seggrid.vectile.c,
                tiles=seggrid.file.pred,
                mosaics=seggrid.vectile.grayscale,
                background=3,
            )
        return files

    @frame.column
    def colored(self) -> pd.Series:
        """
        File-paths to stitched color-coded segmentation masks for each vec-tile.
        Not used in pipeline, but vailable for user's convenience.

        Returns:
            pd.Series: File paths to colored segmentation masks for each vec-tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.file.colored
            xtile  ytile
            9915   12120    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.colored.files(vecgrid)
        self.colored = files
        if not files.map(os.path.exists).all():
            seggrid = vecgrid.seggrid.broadcast
            outgrid = vecgrid

            # preemptively predict so logging appears more sequential
            # else you get "now stitching" before "now predicting"
            _ = seggrid.file.colored

            # only stitch the seggrid which are implicated by the vecgrid
            loc = seggrid.vectile.xtile.isin(outgrid.xtile)
            loc &= seggrid.vectile.ytile.isin(outgrid.ytile)
            seggrid = seggrid.loc[loc]

            vecgrid._stitch_to_file(
                small_grid=seggrid,
                big_grid=outgrid,
                r=seggrid.vectile.r,
                c=seggrid.vectile.c,
                tiles=seggrid.file.colored,
                mosaics=seggrid.vectile.colored,
            )
            assert files.map(os.path.exists).all()

        return files

    @frame.column
    def infile(self) -> pd.Series:
        """
        File-paths to stitched input imagery for each vec-tile.
        Not used in pipeline, but vailable for user's convenience.

        Returns:
            pd.Series: File paths to stitched input imagery for each vec-tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.file.infile
            xtile  ytile
            9915   12120    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.infile.files(vecgrid)
        self.infile = files
        if not files.map(os.path.exists).all():
            seggrid = vecgrid.seggrid
            outgrid = vecgrid

            # preemptively predict so logging appears more sequential
            # else you get "now stitching vectiles" before "now predicting"
            _ = seggrid.file.infile

            # only stitch the seggrid which are implicated by the vecgrid
            loc = seggrid.vectile.xtile.isin(outgrid.xtile)
            loc &= seggrid.vectile.ytile.isin(outgrid.ytile)
            seggrid = seggrid.loc[loc]

            vecgrid._stitch_to_file(
                small_grid=seggrid,
                big_grid=outgrid,
                r=seggrid.vectile.r,
                c=seggrid.vectile.c,
                tiles=seggrid.file.infile,
                mosaics=seggrid.vectile.infile,
            )
            assert (
                seggrid.vectile.infile
                .map(os.path.exists)
                .all()
            )
            assert files.map(os.path.exists).all()

        return files

    @frame.column
    def overlay(self) -> pd.Series:
        """
        File-paths to overlay images showing segmentation masks alpha-blended onto input imagery.
        Not used in pipeline, but vailable for user's convenience.

        Returns:
            pd.Series: File paths to overlay visualization images for each vec-tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.file.overlay
            xtile  ytile
            9915   12120    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.overlay.files(vecgrid)
        self.overlay = files
        if (
                not vecgrid.vectorize
                and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()
        return files

    @frame.column
    def lines(self) -> pd.Series:
        """
        File-paths to line geometry parquet files extracted from segmentation masks.

        Contains centerline network geometries for each vec-tile, representing
        the skeleton of segmented features like sidewalks and crosswalks. These files are
        created during vectorization and used to build the final network output.

        Returns:
            pd.Series: File paths to line geometry parquet files for each vec-tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.file.lines
            xtile  ytile
            9915   12120    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.lines.files(vecgrid)
        self.lines = files
        if (
                not vecgrid.vectorize
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
            >>> ingrid: InGrid
            >>> ingrid.vecgrid.file.polygons
            xtile  ytile
            9915   12120    /home/<user>/tile2net/ma/Boston Common, MA/v...
        """
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.polygons.files(vecgrid)
        self.polygons = files
        if (
                not vecgrid.vectorize
                and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()

        return files

    @frame.column
    def curbs(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.curbs.files(vecgrid)
        self.curbs = files
        if (
                not vecgrid.vectorize
                and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()
        return files

    @frame.column
    def disk_usage(self):
        result = util.path2fsize(self.grayscale)
        result += util.path2fsize(self.colored)
        return result
