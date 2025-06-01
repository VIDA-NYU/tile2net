from __future__ import annotations

from typing import *

if False:
    from .tiles import Tiles


class WithConfig:
    tiles: Tiles

    def from_dict(
            self,
            config: dict[str, Any]
    ) -> Tiles:
        ...

    def from_json(
            self,
            json
    ) -> Tiles:
        ...

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
