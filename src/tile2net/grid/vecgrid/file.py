from __future__ import annotations

import os
import os.path

import pandas as pd

from .. import frame
from ..grid import file
# from ...grid.util import recursion_block

if False:
    from .vecgrid import VecGrid


class File(
    file.File
):
    grid: VecGrid

    @frame.column
    def grayscale(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.tempdir.vecgrid.grayscale.files(vecgrid)
        self.grayscale = files
        if not files.map(os.path.exists).all():
            seggrid = vecgrid.seggrid.broadcast
            outgrid = vecgrid

            # preemptively predict so logging appears more sequential
            # else you get "now stitching" before "now predicting"
            _ = seggrid.file.grayscale

            vecgrid._stitch(
                small_grid=seggrid,
                big_grid=outgrid,
                r=seggrid.vectile.r,
                c=seggrid.vectile.c,
                small_files=seggrid.file.grayscale,
                big_files=seggrid.vectile.grayscale,
                background=3,
            )
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def colored(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.tempdir.vecgrid.colored.files(vecgrid)
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

            vecgrid._stitch(
                small_grid=seggrid,
                big_grid=outgrid,
                r=seggrid.vectile.r,
                c=seggrid.vectile.c,
                small_files=seggrid.file.colored,
                big_files=seggrid.vectile.colored,
            )
            assert files.map(os.path.exists).all()

        return files

    @frame.column
    def infile(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.tempdir.vecgrid.infile.files(vecgrid)
        self.infile = files
        if not files.map(os.path.exists).all():
            seggrid = vecgrid.seggrid
            outgrid = vecgrid

            # preemptively predict so logging appears more sequential
            # else you get "now stitching" before "now predicting"
            _ = seggrid.file.infile

            # only stitch the seggrid which are implicated by the vecgrid
            loc = seggrid.vectile.xtile.isin(outgrid.xtile)
            loc &= seggrid.vectile.ytile.isin(outgrid.ytile)
            seggrid = seggrid.loc[loc]

            vecgrid._stitch(
                small_grid=seggrid,
                big_grid=outgrid,
                r=seggrid.vectile.r,
                c=seggrid.vectile.c,
                small_files=seggrid.file.infile,
                big_files=seggrid.vectile.infile,
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
        vecgrid = self.grid
        files = vecgrid.ingrid.tempdir.vecgrid.overlay.files(vecgrid)
        self.overlay = files
        # if not files.map(os.path.exists).all():
        #     vecgrid._overlay()
        if (
            not vecgrid.vectorize
            and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()
        return files

    @frame.column
    def lines(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.lines.files(vecgrid)
        self.lines = files
        # if not files.map(os.path.exists).all():
        #     vecgrid.vectorize()
        if (
            not vecgrid.vectorize
            and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()
        return files

    @frame.column
    def polygons(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.polygons.files(vecgrid)
        self.polygons = files
        # if not files.map(os.path.exists).all():
        #     vecgrid.vectorize()
        if (
            not vecgrid.vectorize
            and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()

        return files
