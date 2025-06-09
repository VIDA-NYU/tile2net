from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd
import rasterio
from numpy import ndarray

from .tiles import Tiles


def __get__(
        self: Stitched,
        instance: Optional[Tiles],
        owner: type[Tiles],
) -> Stitched:
    if instance is None:
        return self
    try:
        result = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'Tiles must be stitched using `Tiles.stitch` for '
            f'example `Tiles.stitch.to_resolution(2048)` or '
            f'`Tiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.tiles = instance
    return result


class Stitched(
    Tiles
):
    __name__ = 'stitched'
    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(
            self,
            instance: Tiles,
            value: type[Tiles],
    ):
        value.__name__ = self.__name__
        instance.attrs[self.__name__] = value

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        if (
                args
                and callable(args[0])
        ):
            super().__init__(*args[1:], **kwargs)
        else:
            super().__init__(*args, **kwargs)

    @property
    def tiles(self) -> Tiles:
        """Tiles object that this Stitched object is based on"""
        try:
            return self.attrs['tiles']
        except KeyError as e:
            raise AttributeError(

            ) from e

    @tiles.setter
    def tiles(self, value: Tiles):
        """Set the Tiles object that this Stitched object is based on"""
        if not isinstance(value, Tiles):
            raise TypeError(f"Expected Tiles object, got {type(value)}")
        self.attrs['tiles'] = value

    @property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        if 'dimension' in self.attrs:
            return self.attrs['dimension']

        tiles = self.tiles
        dscale = int(self.tscale - tiles.tscale)
        mlength = dscale ** 2
        result = tiles.dimension * mlength
        self.cfg.stitch.dimension = result
        return result

    @property
    def mlength(self):
        tiles = self.tiles
        dscale = int(self.tscale - tiles.tscale)
        return dscale ** 2

    @property
    def group(self) -> pd.Series:
        if 'group' in self.columns:
            return self['group']
        result = pd.Series(np.arange(len(self)), index=self.index)
        self['group'] = result
        return self['group']

    @property
    def outdir(self):
        # return self.tiles.outdir
        tiles = self.tiles
        tiles._stitched = self
        return tiles.outdir

    @property
    def affine_params(self) -> pd.Series:
        key = 'affine_params'
        if key in self:
            return self[key]

        self: pd.DataFrame
        # # it = self['gw gs ge gn'.split()].itertuples(
        #     index=False, name=None
        # )
        # data = [
        #     rasterio.transform
        #     .from_bounds(gw, gs, ge, gn, dim, dim)
        #     for gw, gs, ge, gn in it
        # ]
        dim = self.dimension
        cols = self[['gw', 'gs', 'ge', 'gn']].to_numpy()  # (N, 4) array
        data = [rasterio.transform.from_bounds(l, b, r, t, dim, dim)
                for l, b, r, t in cols]
        result = pd.Series(data, index=self.index, name=key)
        self[key] = result
        return self[key]

    def affine_iterator(self, *args, **kwargs) -> Iterator[ndarray]:
        key = 'affine_iterator'
        cache = self.tiles.attrs
        if key in cache:
            it = cache[key]
        else:
            affine = self.affine_params
            if not self.tiles.cfg.force:
                loc = ~self.tiles.outdir.skip
                affine = affine[loc]

            def gen():
                n = self.cfg.model.bs_val
                a = affine.to_numpy()
                q, r = divmod(len(a), n)
                yield from a[:q * n].reshape(q, n)
                if r:
                    yield a[-r:]

            it = gen()
            cache[key] = it
        yield from it
        del cache[key]

    @property
    def cfg(self):
        return self.tiles.cfg

    @property
    def static(self):
        return self.tiles.static
