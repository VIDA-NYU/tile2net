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
        if (
            not vecgrid._stitch_greyscale
            and not files.map(os.path.exists).all()
        ):
            vecgrid._stitch_greyscale()
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def colored(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.tempdir.vecgrid.colored.files(vecgrid)
        if (
            not vecgrid._stitch_colored
            and not files.map(os.path.exists).all()
        ):
            vecgrid._stitch_colored()
            assert files.map(os.path.exists).all()

        return files

    @frame.column
    def infile(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.tempdir.vecgrid.infile.files(vecgrid)
        self.infile = files
        if (
            not vecgrid._stitch_infile
            and not files.map(os.path.exists).all()
        ):
            vecgrid._stitch_infile()
            assert files.map(os.path.exists).all()

        return files

    @frame.column
    def overlay(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.tempdir.vecgrid.overlay.files(vecgrid)
        if (
                not vecgrid._overlay
                and not files.map(os.path.exists).all()
        ):
            vecgrid._overlay()
        return files

    @frame.column
    def lines(self) -> pd.Series:
        vecgrid = self.grid
        files = vecgrid.ingrid.outdir.vecgrid.lines.files(vecgrid)
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
        if (
                not vecgrid.vectorize
                and not files.map(os.path.exists).all()
        ):
            vecgrid.vectorize()
        return files
