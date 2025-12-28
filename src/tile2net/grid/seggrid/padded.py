from __future__ import annotations

import os
import os.path

import pandas as pd

from .. import frame
from ...grid.frame.namespace import namespace

if False:
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
        Number of InGrid tiles that comprise one dimension of a seg-tile
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
        return self.instance.ingrid.dimension * self.length

    @property
    def grid(self) -> SegGrid:
        return self.instance

    @frame.column
    def infile(self) -> pd.Series:
        """
        Padded input imagery file for each seg-tile.
        Stitches input files when seggrid.file is accessed.
        """
        seggrid = self.grid
        files = seggrid.ingrid.outdir.seggrid.padded.infile.files(seggrid)

        self.infile = files
        if not files.map(os.path.exists).all():
            ingrid = seggrid.ingrid.broadcast
            small_files = ingrid.file.infile
            big_files = ingrid.segtile.infile
            assert (
                ingrid.file.infile
                .map(os.path.exists)
                .all()
            )
            ingrid._stitch(
                small_grid=ingrid,
                big_grid=seggrid,
                r=ingrid.segtile.r,
                c=ingrid.segtile.c,
                small_files= small_files,
                big_files= big_files,
            )
            msg = f"Files not stitched: {files[~files.map(os.path.exists)]}"
            assert files.map(os.path.exists).all(), msg

        return files
