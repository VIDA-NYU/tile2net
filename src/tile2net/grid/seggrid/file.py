from __future__ import annotations

import hashlib
import os
import sys

import pandas as pd

from .postprocess import PostProcess
from .. import frame
from ..grid import file

from ...grid import util

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))

if False:
    from .seggrid import SegGrid


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


class File(
    file.File
):
    grid: SegGrid

    @PostProcess
    def postprocess(self) -> pd.Series:
        """
        Namespace for work-in-progress postprocessing of segmentation results.
        """

    @frame.column
    def infile(self) -> pd.Series:
        """
        A file for each seg-tile: the stitched input grid.
        Stitches input files when seggrid.file is accessed
        """
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.infile.files(grid)
        self.infile = files
        if not files.map(os.path.exists).all():
            ingrid = grid.ingrid
            _ = ingrid.file.infile
            mosaics = ingrid.segtile.infile
            assert (
                ingrid.file.infile
                .map(os.path.exists)
                .all()
            )
            ingrid._stitch_to_file(
                small_grid=ingrid,
                big_grid=grid,
                r=ingrid.segtile.r,
                c=ingrid.segtile.c,
                tiles=ingrid.file.infile,
                mosaics=mosaics,
            )
            msg = f"Files not stitched: {files[~files.map(os.path.exists)]}"
            assert files.map(os.path.exists).all(), msg

        return files

    @frame.column
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.pred
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.pred.files(grid)
        self.pred = files
        if (
                bool(grid.predict)
                and not files.map(os.path.exists).all()
        ):
            grid.file.pred = files
            grid.predict(probs=False)
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def prob(self) -> pd.Series:
        # problem is, we also need file.pred
        grid = self.grid.broadcast
        files = grid.ingrid.outdir.seggrid.prob.files(grid)
        self.prob = files
        if (
                bool(self.grid.predict)
                and not files.map(os.path.exists).all()
        ):
            predict = getattr(grid, 'predict')
            grid.file.prob = files
            setattr(grid, 'predict', False)

            _ = grid.file.pred
            # grid.predict = predict
            setattr(grid, 'predict', predict)
            grid.predict(probs=True)

            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.pred
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid.broadcast
        files = grid.outdir.seggrid.pred.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.pred = files
        self.pred = files
        if (
                not grid.predict
                and not files.map(os.path.exists).all()
        ):
            grid.predict(probs=False)
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def prob(self) -> pd.Series:
        grid = self.grid.broadcast
        files = grid.outdir.seggrid.prob.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.prob = files
        self.prob = files
        if (
            not bool(grid.predict)
            and not files.map(os.path.exists).all()
        ):
            grid.predict(probs=True)
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def error(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.error.files(grid)
        self.error = files
        if (
                grid.predict
                and not files.map(os.path.exists).all()
        ):
            grid.file.error = files
            grid.predict(probs=True)
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def colored(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.colored
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.colored.files(grid)
        self.colored = files
        if (
                grid.predict
                and not files.map(os.path.exists).all()
        ):
            grid.file.colored = files
            grid.predict()
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def intensity(self) -> pd.Series:
        """
        File-paths to intensity representation of segmentation results.

        Alternative visualization format where segmentation classes are represented
        as intensity values.

        Returns:
            pd.Series: File paths to intensity representations for each seg-tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.intensity
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.intensity.files(grid)
        self.intensity = files
        if (
                grid.predict
                and not files.map(os.path.exists).all()
        ):
            grid.file.intensity = files
            grid.predict()
            assert files.map(os.path.exists).all()
        return files

    def output(self, dirname: str) -> pd.Series:
        # Note: Can't use frame.column with parameters
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.output.files(grid, dirname)
        key = f'output.{dirname}'
        setattr(self, key, files)
        # if not files.map(os.path.exists).all():
        #     grid.predict()
        if (
                grid.predict
                and not files.map(os.path.exists).all()
        ):
            grid.predict()
        assert files.map(os.path.exists).all()
        return files

    @frame.column
    def disk_usage(self):
        result = util.path2fsize(self.pred)
        result += util.path2fsize(self.colored)
        return result
