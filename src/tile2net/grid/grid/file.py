from __future__ import annotations
from pathlib import Path
import contextlib

import os

import pandas as pd

from tile2net.grid.source import Local, Remote
from tile2net.logger import logger

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
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.static = files
        setattr(self, 'static', files)

        if (
            isinstance(source, Remote)
            and self
        ):
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            grid.outdir
            .__getattribute__('source')
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        n = loc.sum()
        if (
            isinstance(source, Remote)
            and loc.any()
        ):
            msg = f'{trace} found {n} missing files. Downloading to\n\t{path}'
            logger.info(msg)
            source.download()
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
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
        setattr(self, 'pred', files)
        if self:
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            grid.outdir
            .__getattribute__('project')
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Unstitching to\n\t{path}'
            logger.info(msg)
            subset = grid.loc[loc]
            subset._unstitch2file(
                tiles=subset.file.pred,
                mosaics=subset.segtile.pred,
                row=subset.segtile.row,
                col=subset.segtile.col,
            )
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @frame.column
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.
        """
        grid = self.basegrid
        files = grid.outdir.project.prob.files(grid)
        setattr(self, 'prob', files)
        if self:
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            grid.outdir
            .__getattribute__('project')
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Unstitching to\n\t{path}'
            logger.info(msg)
            subset = grid.loc[loc]
            subset._unstitch2file(
                tiles=subset.file.prob,
                mosaics=subset.segtile.prob,
                row=subset.segtile.row,
                col=subset.segtile.col,
            )
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @frame.property
    def network(self):
        file = self.basegrid.outdir.project.network.parquet
        setattr(self, 'network', file)
        if not self:
            _ = self.basegrid.network
        return file

    @frame.property
    def polygons(self):
        file = self.basegrid.outdir.project.polygons.parquet
        setattr(self, 'polygons', file)
        if not self:
            _ = self.basegrid.polygons
        return file
