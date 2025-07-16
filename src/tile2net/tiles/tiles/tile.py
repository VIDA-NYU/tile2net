from __future__ import annotations

import copy
import functools

import pandas as pd

import tile2net.tile

if False:
    from tile2net.tiles.tiles.tiles import Tiles


def __get__(
        self: Indexer,
        instance: Tile,
        owner
):
    self.tile = instance
    self.tiles = instance.tiles
    return copy.copy(self)


class Indexer:
    tile: Tile = None
    tiles: Tiles = None
    locals().update(
        __get__=__get__,
    )

    def __init__( self, *args, ):
        ...


class ILoc(Indexer):

    def __getitem__(self, item) -> tile2net.tile.Tile | pd.Series:
        tiles = self.tile.tiles
        ndframe = tiles.iloc[item]
        if isinstance(ndframe, pd.Series):
            ndframe = tiles.iloc[[item]]
            result = tile2net.tile.Tile(
                tuple=next(ndframe.itertuples()),
                tiles=self.tile.tiles,
            )
        else:
            data = [
                tile2net.tile.Tile(
                    tuple=row,
                    tiles=self.tile.tiles,
                )
                for row in ndframe.itertuples()
            ]
            result = pd.Series( data, index=ndframe.index )

        return result

class Loc(Indexer):
    def __getitem__(self, item) -> tile2net.tile.Tile | pd.Series:
        tiles = self.tile.tiles
        ndframe = tiles.loc[item]
        if isinstance(ndframe, pd.Series):
            ndframe = tiles.loc[[item]]
            result = tile2net.tile.Tile(
                tuple=next(ndframe.itertuples()),
                tiles=self.tile.tiles,
            )
        else:
            data = [
                tile2net.tile.Tile(
                    tuple=row,
                    tiles=self.tile.tiles,
                )
                for row in ndframe.itertuples()
            ]
            result = pd.Series(data, index=ndframe.index)

        return result


class cached_property:
    __wrapped__ = None
    instance = None

    def __init__(
            self,
            func
    ):
        functools.update_wrapper(self, func)

    def __set_name__(self, owner, name):
        self.__name__ = name
        self.owner = owner

    @functools.cached_property
    def key(self):
        # from .tiles import Tiles
        # if issubclass(self.owner, Tiles):
        if issubclass(self.owner, pd.DataFrame):
            return self.__name__
        elif hasattr(self.owner, 'tiles'):
            return f'{self.instance.__name__}.{self.__name__}'
        else:
            raise TypeError(
                f'Owner {self.owner} must be a Tiles subclass or have a tiles attribute.'
            )

    @property
    def cache(self) -> dict:
        if issubclass(self.owner, pd.DataFrame):
            cache = self.instance.attrs
        elif hasattr(self.owner, 'tiles'):
            cache = self.instance.tiles.attrs
        else:
            raise TypeError(
                f'Owner {self.owner} must be a Tiles subclass or have a tiles attribute.'
            )
        return cache

    def __get__(
            self,
            instance,
            owner
    ):
        self.instance = instance
        if instance is None:
            return self
        key = self.key

        cache = self.cache
        if key in cache:
            return cache[key]
        result = self.__wrapped__(instance)
        cache[key] = result
        return result

    def __set__(self, instance, value):
        self.instance = instance
        key = self.key
        cache = self.cache
        cache[key] = value

    def __delete__(self, instance):
        self.instance = instance
        key = self.key
        cache = self.cache
        cache.pop(key, None)


def __get__(
        self: Tile,
        instance: Tiles,
        owner: type[Tiles],
) -> Tile:
    self.tiles = instance
    return copy.copy(self)


class sticky:
    cached_property = cached_property


class static:
    class cached_prpoerty(cached_property):


        @property
        def cache(self) -> dict:
            if issubclass(self.owner, pd.DataFrame):
                cache = self.instance.__dict__
            elif hasattr(self.owner, 'tiles'):
                cache = self.instance.tiles.__dict__
            else:
                raise TypeError(
                    f'Owner {self.owner} must be a Tiles subclass or have a tiles attribute.'
                )
            return cache





class Tile(

):
    locals().update(
        __get__=__get__
    )
    tiles: Tiles = None

    @property
    def intiles(self):
        return self.tiles.intiles

    @property
    def vectiles(self):
        return self.tiles.vectiles

    @property
    def segtiles(self):
        return self.tiles.segtiles

    @property
    def zoom(self):
        return self.intiles.tile.zoom

    @cached_property
    def scale(self) -> int:
        """
        Tile scale; the XYZ scale of the tiles.
        Higher value means smaller area.
        """

    @cached_property
    def dimension(self) -> int:
        """Tile dimension; inferred from input files"""
        return self.intiles.tile.dimension * self.length

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dimension, self.dimension, 3

    @cached_property
    def length(self) -> int:
        """How many input tiles comprise a tile of this class"""
        raise NotImplemented

    @cached_property
    def area(self):
        return self.length ** 2

    def __init__(self, *args):
        ...

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __iter__(self):
        ...

    @ILoc
    def iloc(self):
        ...

    @Loc
    def loc(self):
        ...

