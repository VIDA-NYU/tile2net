from __future__ import annotations

from typing import *
import os
import os.path

import pandas as pd
from tile2net.core.frame import frame
from tile2net.core.frame.namespace import namespace

if TYPE_CHECKING:
    from .seggrid import SegGrid


class Padded(
    namespace
):
    """
    See usage:
        >>> SegGrid.padded
    """

    instance: SegGrid
    @property
    def length(self) -> int:
        """
        Number of Grid tiles that comprise one dimension of a seg-tile
        after it has been padded a number of in-tiles.
        """
        result = self.instance.length
        result += self.instance.cfg.segmentation.pad * 2
        return result

    @property
    def dimension(self) -> int:
        """
        Pixel dimension of each seg-tile after it has been
        padded a number of in-tiles.
        """
        out = self.instance.length * self.instance.ingrid.dimension
        out += 2 * self.instance.cfg.segmentation.pad
        return out

    @property
    def basegrid(self) -> SegGrid:
        return self.instance

    @frame.column
    def static(self) -> pd.Series:
        """
        Padded input imagery file for each seg-tile.
        Stitches input files when seggrid.file is accessed.
        """
        seggrid = self.basegrid
        files = seggrid.ingrid.outdir.seggrid.padded.static.files(seggrid)

        self.static = files
        if not files.map(os.path.exists).all():
            grid = seggrid.ingrid.broadcast
            small_files = grid.file.static
            big_files = grid.segtile.static
            assert (
                grid.file.static
                .map(os.path.exists)
                .all()
            )
            grid._stitch2file(
                row=grid.segtile.row,
                col=grid.segtile.col,
                tiles= small_files,
                mosaics= big_files,
            )
            msg = f"Files not stitched: {files[~files.map(os.path.exists)]}"
            assert files.map(os.path.exists).all(), msg

        return files
