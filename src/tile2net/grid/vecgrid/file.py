from __future__ import annotations

import os
import os.path
from typing import *

import pandas as pd

from tile2net.logger import logger
from .. import frame, util
from ..basegrid import file

if TYPE_CHECKING:
    from .vecgrid import VecGrid
    from ..grid import Grid


class File(
    file.File
):
    basegrid: VecGrid

    @property
    def static(self) -> pd.Series:
        """
        A file for each seg-tile: the stitched input grid.
        Stitches input files when seggrid.file is accessed
        """
        return self.basegrid.seggrid.file.static

    @property
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.
        # TODO: update

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        """
        return self.basegrid.seggrid.file.pred

    @property
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.
        # TODO: update
        """
        return self.basegrid.seggrid.file.prob

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
        setattr(self, 'network', files)
        if (
            self
            or bool(vecgrid.vectorize)
        ):
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            vecgrid.grid.outdir
            .__getattribute__(vecgrid.__name__)
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Vectorizing to\n\t{path}'
            logger.info(msg)
            vecgrid.vectorize()
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
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
        setattr(self, 'polygons', files)
        if (
            self
            or bool(vecgrid.vectorize)
        ):
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            vecgrid.grid.outdir
            .__getattribute__(vecgrid.__name__)
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Vectorizing to\n\t{path}'
            logger.info(msg)
            vecgrid.vectorize()
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @frame.column
    def curbs(self) -> pd.Series:
        # todo: needs documentation
        vecgrid = self.basegrid
        files = vecgrid.grid.outdir.vecgrid.curbs.files(vecgrid)
        setattr(self, 'curbs', files)
        if (
            self
            or bool(vecgrid.vectorize)
        ):
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            vecgrid.grid.outdir
            .__getattribute__(vecgrid.__name__)
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Vectorizing to\n\t{path}'
            logger.info(msg)
            vecgrid.vectorize()
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
        return files

    @property
    def mask(self) -> pd.Series:
        """
        A file for each vec-tile: the stitched mask from seg-tiles.
        Stitches mask files when vecgrid.file.mask is accessed.
        """
        return self.basegrid.seggrid.file.mask

    @frame.column
    def disk_usage(self):
        # todo: include other files
        result = util.path2fsize(self.grayscale)
        result += util.path2fsize(self.colorized)
        return result
