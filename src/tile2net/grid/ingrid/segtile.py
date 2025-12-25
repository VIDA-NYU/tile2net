from __future__ import annotations

import numpy as np

from .. import frame

import copy

import pandas as pd

from tile2net.grid.frame.namespace import namespace

if False:
    from .ingrid import InGrid


class SegTile(
    namespace,
):
    ingrid: InGrid

    @property
    def ingrid(self) -> InGrid:
        return self.instance

    @property
    def grid(self) -> InGrid:
        return self.ingrid

    @property
    def length(self):
        """
        Number of InGrid tiles in one dimension of the segmentation tile.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.length
            4
        """
        return self.ingrid.seggrid.length

    @property
    def dimension(self):
        """
        Pixel dimension of each segmentation tile.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.dimension
            1024
        """
        return self.ingrid.dimension * self.length

    # @property
    # def shape(self) -> tuple[int, int, int]:
    #     return self.dimension, self.dimension, self.ingrid.shape[2]

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of each segmentation tile as (height, width).

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.shape
            (1024, 1024)
        """
        return self.dimension, self.dimension

    @property
    def index(self):
        """
        MultiIndex of (xtile, ytile) for segmentation tiles.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.index
            MultiIndex([(79320, 96960)], names=['segtile.xtile', 'segtile.ytile'])
        """
        arrays = self.xtile, self.ytile
        names = self.xtile.name, self.ytile.name
        result = pd.MultiIndex.from_arrays(arrays, names=names)
        return result

    @frame.column
    def itile(self):
        """
        Integer identifier for each segmentation tile.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.itile
            xtile   ytile
            317280  387840    123456
            Name: segtile.itile, dtype: int64
        """
        seggrid = self.grid.seggrid.broadcast
        result = (
            self.grid.seggrid.broadcast.itile
            .loc[~seggrid.index.duplicated()]
            .loc[self.index]
            .values
        )
        return result

    @frame.column
    def xtile(self):
        """
        X tile id of the SegGrid tile associated with the InGrid tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.xtile
            xtile   ytile
            317280  387840    79320
            Name: segtile.xtile, dtype: int64
        """
        ingrid = self.ingrid

        seggrid = ingrid.seggrid.filled
        result: pd.Index = ingrid.xtile.__floordiv__(ingrid.segtile.length)

        msg = 'All segtile.xtile must be in seggrid.xtile!'
        assert result.isin(seggrid.xtile).all(), msg
        return result

    @frame.column
    def ytile(self):
        """
        Y tile id of the SegGrid tile associated with the InGrid tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.ytile
            xtile   ytile
            317280  387840    96960
            Name: segtile.ytile, dtype: int64
        """
        ingrid = self.ingrid

        seggrid = ingrid.seggrid.filled
        result: pd.Index = ingrid.ytile.__floordiv__(ingrid.segtile.length)

        msg = 'All segtile.ytile must be in seggrid.ytile!'
        assert result.isin(seggrid.ytile).all(), msg
        return result

    @frame.column
    def r(self) -> pd.Series:
        """
        Row index within the segmentation tile (0 to length-1).

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.r
            xtile   ytile
            317280  387840    0
            Name: segtile.r, dtype: int64
        """
        ingrid = self.ingrid
        result = (
            ingrid.ytile
            .to_series(index=ingrid.index)
            .floordiv(ingrid.segtile.length)
            .mul(ingrid.segtile.length)
            .rsub(ingrid.ytile.values)
        )
        return result

    @frame.column
    def c(self) -> pd.Series:
        """
        Column index within the segmentation tile (0 to length-1).

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.c
            xtile   ytile
            317280  387840    0
            Name: segtile.c, dtype: int64
        """
        ingrid = self.ingrid
        result = (
            ingrid.xtile
            .to_series(index=ingrid.index)
            .floordiv(ingrid.segtile.length)
            .mul(ingrid.segtile.length)
            .rsub(ingrid.xtile.values)
        )
        return result

    @frame.column
    def infile(self) -> pd.Series:
        """
        Path to input file for this segmentation tile.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.infile
            xtile   ytile
            317280  387840    /home/.../ma/Boston Common, MA/s...
            Name: segtile.infile, dtype: object
        """
        ingrid = self.ingrid
        result = (
            # ingrid.seggrid.file.stitched
            ingrid.seggrid.filled.file.infile
            .loc[self.index]
            .values
        )
        return result

    # @frame.column
    # def polygon(self) -> pd.Series:
    #     """seggrid.file broadcasted to ingrid"""
    #     ingrid = self.ingrid
    #     result = (
    #         ingrid.seggrid.filled.file
    #         .loc[self.index]
    #         .values
    #     )
    #     return result
    #

    @frame.column
    def grayscale(self) -> pd.Series:
        """
        Path to grayscale segmentation file for this tile.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.grayscale
            xtile   ytile
            317280  387840    /home/.../ma/Boston Common, MA/s...
            Name: segtile.grayscale, dtype: object
        """
        ingrid = self.ingrid
        seggrid = ingrid.seggrid.broadcast
        result = (
            seggrid.file.grayscale
            .loc[~seggrid.index.duplicated()]
            .loc[self.index]
            .values
        )
        return result


    @property
    def pad(self) -> int:
        """
        Padding pixels for segmentation tiles.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.segtile.pad
            64
        """
        return self.grid.cfg.segmentation.pad

    @pad.setter
    def pad(self, value: int) -> None:
        self.grid.cfg.segmentation.pad = value

    @pad.deleter
    def pad(self) -> None:
        del self.grid.cfg.segmentation.pad



