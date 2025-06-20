from __future__ import annotations

import copy
from functools import *
from pathlib import Path

import gdown
from PIL import Image

if False:
    from tile2net.tiles.tiles import Tiles


def __get__(
        self: Static,
        instance: Tiles,
        owner: type[Tiles]
) -> Static:
    self.tiles = instance
    return copy.copy(self)


class Static:
    tiles: Tiles
    locals().update(__get__=__get__)

    def __init__(self, *args, ):
        ...

    """Static directory into which weights are saved."""
    path = Path(__file__).parent
    while path.name != 'tile2net':
        path = path.parent
        if not path.name:
            raise FileNotFoundError('Could not find tile2net directory')
    path = path / 'static'

    @classmethod
    def download(self):
        url = 'https://drive.google.com/drive/folders/1cu-MATHgekWUYqj9TFr12utl6VB-XKSu'
        gdown.download_folder(
            url=url,
            # quiet=True,
            output=self.path.__str__(),
        )

    hrnet_checkpoint = (
        path
        .joinpath('hrnetv2_w48_imagenet_pretrained.pth')
        .absolute().__fspath__()
    )

    snapshot = (
        path
        .joinpath('satellite_2021.pth')
        .absolute().__fspath__()
    )

    @cached_property
    def black(self) -> Path:
        dim: int = self.tiles.tile.dimension
        path: Path = self.path.joinpath(str(dim), 'black.png')

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (dim, dim), (0, 0, 0)).save(path)

        return path
