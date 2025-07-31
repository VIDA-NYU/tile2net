from __future__ import annotations

from typing import *

if False:
    from .grid import Tiles


class WithConfig:
    grid: Tiles

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
        self.grid = instance
        self.Tiles = owner
        return self

    def __init__(self, *args, **kwargs):
        ...
