from __future__ import annotations

import copy
import functools
from pathlib import Path

import imageio.v3 as iio

if False:
    from tile2net.tiles.tiles.tiles import Tiles


def __get__(
        self: File,
        instance: Tiles,
        owner: type[Tiles],
) -> File:
    from .tiles import Tiles
    self.tiles = instance
    return copy.copy(self)


class File(

):
    locals().update(
        __get__=__get__
    )
    tiles: Tiles = None

    def __init__(self, *args):
        ...
