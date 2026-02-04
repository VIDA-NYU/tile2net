from __future__ import annotations

from typing import *

from pathlib import Path
import contextlib

import os

import pandas as pd

from tile2net.core.source import Local, Remote
from tile2net.logger import logger

if TYPE_CHECKING:
    from tile2net.core.frame import column
    from tile2net.core.ingrid import InGrid
    from tile2net.core.frame import column
    from tile2net.core.ingrid import InGrid

from .. import frame
from tile2net.core.grid import file

if TYPE_CHECKING:
    from .ingrid import InGrid


class File(
    file.File
):
    instance: InGrid
    grid: InGrid

    @frame.column
    def static(self):
        """Static imagery from the source."""
        grid = self.grid
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

    @frame.column
    def mask(self):
        mask = self.grid.mask
        out = mask.files()
        return out

    @contextlib.contextmanager
    def _static_peek(self):
        key = self.grid.__class__.file.static.key
        grid = self.grid
        delete = key not in grid.columns
        with self:
            yield
        if delete:
            del grid[key]

    @frame.property
    def sample(self) -> str:
        """A sample filepath from static; a singular saved file instead of a whole Series"""
        grid = self.grid
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

    @property
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.
        """
        return self.grid.seggrid.file.pred

    @property
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.
        """
        return self.grid.seggrid.file.prob

    @frame.property
    def network(self):
        file = self.grid.outdir.project.network.parquet
        setattr(self, 'network', file)
        if not self:
            _ = self.grid.network
        return file

    @frame.property
    def polygons(self):
        file = self.grid.outdir.project.polygons.parquet
        setattr(self, 'polygons', file)
        if not self:
            _ = self.grid.polygons
        return file
