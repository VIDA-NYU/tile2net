from __future__ import annotations

from dataclasses import dataclass
from typing import *

from . import cached

if False:
    from ..tiles.tiles.tiles import Tiles


class File:

    @cached.property
    def greyscale(self) -> str:
        ...

    @cached.property
    def colored(self) -> str:
        ...

    def __init__(self, *args):
        ...


@dataclass
class Tile:
    tuple: Any
    tiles: Tiles

    @cached.property
    def xtile(self) -> int:
        return self.tuple.Index[0]

    @cached.property
    def ytile(self) -> int:
        return self.tuple.Index[1]

    @cached.property
    def scale(self) -> int:
        return self.tuple.scale

    @cached.property
    def dimension(self) -> int:
        return self.tuple.dimension

    @cached.property
    def shape(self) -> tuple[int, int, int]:
        return self.tuple.shape

    @File
    def file(self):
        """Nested namespace for file-related properties."""
        # This code block does not run, it is syntactic sugar. See:
        greyscale = self.file.greyscale
        colored = self.file.colored

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}(\n"
               f"xtile={self.xtile},\n"
                f"ytile={self.ytile},\n"
                f"scale={self.scale},\n"
                f"dimension={self.dimension},\n"
                f"shape={self.shape},\n"
            ")"
        )



