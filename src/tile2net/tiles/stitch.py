from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import *

if False:
    from .tiles import Tiles


class Stitch:
    tiles: Tiles

    def to_resolution(
            self,
            resolution: int = 1024,
            pad: bool = True,
    ) -> Tiles:
        """
        resolution:
            target resolution of the stitched tiles
        pad:
            True:
                Go back and include more tiles to ensure the stitched
                tiles include all the original tiles.
            False:
                Drop any tiles that cannot be stitched into a complete
                tile.

        Returns:
            Tiles:
                New Tiles object with Tiles.stitched set to the stitched
                tiles at the specified resolution.
        """
        tiles = self.tiles
        try:
            _ = tiles.resolution
        except KeyError as e:
            msg = (
                'You cannot stitch tiles to a resolution when the tile '
                'resolution has not yet been set. First you must perform'
                ' tiles.with_indir() or tiles.with_source() for the '
                'resolution to be determined.'
            )
            raise ValueError(msg) from e
        dscale = int(math.log2(resolution / tiles.resolution))
        scale = tiles.tscale - dscale
        result = tiles.stitch.to_scale(scale, pad=pad)
        return result

    def to_cluster(
            self,
            cluster: int = 16,
            pad: bool = True,
    ) -> Tiles:
        """
        cluster:
            Cluster size of the stitched tiles. Must be a power of 2.
        pad:
            True:
                Go back and include more tiles to ensure the stitched
                tiles include all the original tiles.
            False:
                Drop any tiles that cannot be stitched into a complete
                tile.

        Returns:
            Tiles:
                New Tiles object with Tiles.stitched set to the stitched
                tiles at the specified cluster size.
        """
        if (
                not isinstance(cluster, int)
                or cluster <= 0
                or (cluster & (cluster - 1)) != 0
        ):
            raise ValueError('Cluster must be a positive power of 2.')
        tiles = self.tiles
        dscale = int(math.log2(cluster))
        scale = tiles.tscale - dscale
        result = tiles.stitch.to_scale(scale, pad=pad)
        return result

    # def to_scale(
    #         self,
    #         scale: int = 17,
    #         pad: bool = True,
    # ) -> Tiles:
    #     """
    #     scale:
    #         Scale of larger slippy tiles into which smaller tiles are stitched
    #     pad:
    #         True:
    #             Go back and include more tiles to ensure the stitched
    #             tiles include all the original tiles.
    #         False:
    #             Drop any tiles that cannot be stitched into a complete
    #             tile.
    #
    #     Returns:
    #         Tiles:
    #             New Tiles object with Tiles.stitched set to the stitched
    #             tiles at the specified scale.
    #     """
    #     from .tiles import Tiles
    #     TILES = self.tiles
    #
    #     tiles = TILES
    #
    #     dscale = tiles.tscale - scale
    #     # todo: how to get the bounds in a different scale
    #     if dscale < 0:
    #         msg = (
    #             f'Cannot stitch from slippy scale {tiles.tscale} to '
    #             f'slippy scale {scale}. The target must be a lower int.'
    #         )
    #         raise ValueError(msg)
    #     ntiles = dscale ** 2
    #
    #     txy = tiles.index
    #     tx = txy.get_level_values(0)
    #     ty = txy.get_level_values(1)
    #     mx = tx // ntiles
    #     my = ty // ntiles
    #     mxy = pd.MultiIndex.from_arrays([mx, my])
    #
    #     assert not txy.has_duplicates, 'Duplicate tiles!'
    #
    #     if pad:
    #         # group: pd.MultiIndex = mxy.unique() * ntiles
    #         unique = mxy.unique()
    #         mx = unique.get_level_values(0) * ntiles
    #         my = unique.get_level_values(1) * ntiles
    #         group = pd.MultiIndex.from_arrays([mx, my])
    #
    #         arange = np.arange(ntiles)
    #         rc = pd.MultiIndex.from_product([arange, arange])
    #         rc = rc.repeat(len(group))
    #         group = group.repeat(ntiles * ntiles)
    #         index = group + rc
    #         ix = index.get_level_values(0)
    #         iy = index.get_level_values(1)
    #         _tiles = tiles
    #         tiles = Tiles.from_integers(ix, iy, zoom=tiles.zoom)
    #         tiles.attrs.update(tiles.attrs)
    #     else:
    #         loc = (
    #             pd.Series(1, index=mxy)
    #             .groupby(mxy)
    #             .sum()
    #             .eq(ntiles * ntiles)
    #             .loc[mxy]
    #         )
    #         tiles = tiles.loc[loc]
    #
    #     txy = tiles.index
    #     mxy: pd.MultiIndex = txy // ntiles
    #     mx = mxy.get_level_values(0)
    #     my = mxy.get_level_values(1)
    #     rc = txy % ntiles
    #     r = rc.get_level_values(0)
    #     c = rc.get_level_values(1)
    #     px = r * tiles.resolution
    #     py = c * tiles.resolution
    #
    #     tiles = tiles.assign(
    #         mx=mx,
    #         my=my,
    #         r=r,
    #         c=c,
    #         px=px,
    #         py=py,
    #     )
    #
    #     index = tiles.index.unique()
    #     tx: pd.Index = index.get_level_values(0)
    #     ty: pd.Index = index.get_level_values(1)
    #     stitched = Tiles.from_integers(tx=tx, ty=ty, zoom=scale)
    #
    #     if (
    #             tiles.attrs.get('indir')
    #     ):
    #         raise NotImplementedError('todo: actually stitch the files')
    #
    #     tiles: Tiles
    #     tiles.stitched = stitched
    #
    #     return tiles

    def to_scale(
            self,
            scale: int = 17,
            pad: bool = True,
    ) -> Tiles:
        """
        scale:
            Scale of larger slippy tiles into which smaller tiles are stitched
        pad:
            True:
                Go back and include more tiles to ensure the stitched
                tiles include all the original tiles.
            False:
                Drop any tiles that cannot be stitched into a complete
                tile.
        """
        from .tiles import Tiles
        TILES = self.tiles

        tiles = TILES

        dscale = tiles.tscale - scale
        # todo: how to get the bounds in a different scale
        if dscale < 0:
            msg = (
                f'Cannot stitch from slippy scale {tiles.tscale} to '
                f'slippy scale {scale}. The target must be a lower int.'
            )
            raise ValueError(msg)
        mlength = dscale ** 2  # mosaic length
        marea = dscale ** 4 # mosaic area

        txy = tiles.index.to_frame()
        if txy.duplicated().any():
            msg = ('Cannot stitch tiles with duplicate indices!')
            raise ValueError(msg)

        # a mosaic is a group of tiles
        if pad:
            # fill all implicated mosaics
            mxy: pd.DataFrame = txy // mlength
            topleft = (
                mxy
                .drop_duplicates()
                .mul(mlength)
            )

            arange = np.arange(mlength)
            x, y = np.meshgrid(arange, arange, indexing='ij')
            txy: np.ndarray = (
                np.stack((x, y), -1)
                .reshape(-1, 2)
                .__add__(topleft.values[:, None, :])
                .reshape(-1, 2)
            )
            tx = txy[:, 0]
            ty = txy[:, 1]
            assert not pd.DataFrame(dict(tx=tx, ty=ty)).duplicated().any()
            tiles = Tiles.from_integers(tx, ty, zoom=tiles.zoom)
            tiles.attrs.update(TILES.attrs)
        else:
            # drop mosaics not containing all tiles
            mxy: pd.DataFrame = txy // mlength
            loc = (
                pd.Series(1, index=mxy)
                .groupby(mxy)
                .sum()
                .eq(mlength * mlength)
                .loc[mxy]
            )
            tiles = tiles.loc[loc]

        mxy = (
            tiles.index
            .to_frame()
            .floordiv(mlength)
        )
        unique = mxy.drop_duplicates()
        tx = unique.tx
        ty = unique.ty
        stitched = Tiles.from_integers(tx=tx, ty=ty, zoom=scale)
        stitched.zoom = tiles.zoom
        tiles.stitched = stitched

        msg = f'Tile count is not {marea}x the mosaic count.'
        assert len(tiles) == len(tiles.stitched) * marea, msg
        msg = f'Not all mosaics are complete'
        assert (
            tiles
            .groupby(pd.MultiIndex.from_frame(mxy))
            .size()
            .eq(marea)
            .all()
        ), msg


        if (
                tiles.attrs.get('indir')
        ):
            raise NotImplementedError('todo: actually stitch the files')

        return tiles

    def __get__(
            self,
            instance,
            owner
    ) -> Self:
        self.tiles = instance
        self.Tiles = owner
        return self

    def __init__(self, *args, **kwargs):
        ...
