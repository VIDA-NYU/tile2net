from __future__ import annotations

from typing import *

import os
import os.path

import pandas as pd

from .. import frame, util
from ..basegrid import file
from tile2net.logger import logger

if TYPE_CHECKING:
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
        files = self.dir.static.files(grid)
        setattr(self, 'static', files)
        if self:
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            self.dir
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Stitching to\n\t{path}'
            logger.info(msg)
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.static.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.static,
                mosaics=seggrid.vectile.static,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
            )
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
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
        files = self.dir.pred.files(grid)
        setattr(self, 'pred', files)
        if self:
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            self.dir
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Stitching to\n\t{path}'
            logger.info(msg)
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.pred.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.pred,
                mosaics=seggrid.vectile.pred,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
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
        # TODO: update
        """
        grid = self.basegrid
        files = self.dir.prob.files(grid)
        setattr(self, 'prob', files)
        if self:
            return files

        name = (
            str(files.name)
            .rsplit('.', 1)[-1]
        )
        path: str = (
            self.dir
            .__getattribute__(name)
            .dir
        )
        trace = f'{self._trace}.{name}'

        loc = ~files.map(os.path.exists)
        if loc.any():
            n = loc.sum()
            msg = f'{trace} found {n} missing files. Stitching to\n\t{path}'
            logger.info(msg)
            seggrid = grid.seggrid
            loc = ~seggrid.vectile.prob.map(os.path.exists)
            seggrid = seggrid.loc[loc]
            seggrid._stitch2file(
                tiles=seggrid.file.prob,
                mosaics=seggrid.vectile.prob,
                row=seggrid.vectile.row,
                col=seggrid.vectile.col,
            )
            assert files.map(os.path.exists).all()
        else:
            msg = f'{trace} found all {len(loc)} files already in \n\t{path}'
            logger.info(msg)
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
        setattr(self, 'lines', files)
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

    @frame.column
    def disk_usage(self):
        # todo: include other files
        result = util.path2fsize(self.grayscale)
        result += util.path2fsize(self.colorized)
        return result
