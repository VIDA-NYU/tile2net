from __future__ import annotations

from functools import *
from pathlib import Path
from typing import *

import gdown
from PIL import Image

if False:
    from .intiles import InTiles


class Static:
    intiles: InTiles

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    def __get__(
            self,
            intiles: InTiles,
            owner: type[InTiles]
    ):
        self.intiles = intiles
        return self

    @cached_property
    def path(self) -> Path:
        """Static directory into which weights are saved."""
        path = Path(__file__).parent
        while path.name != 'tile2net':
            path = path.parent
            if not path.name:
                raise FileNotFoundError('Could not find tile2net directory')
        path = path / 'static'
        return path

    def download(self):
        url = 'https://drive.google.com/drive/folders/1cu-MATHgekWUYqj9TFr12utl6VB-XKSu'
        gdown.download_folder(
            url=url,
            # quiet=True,
            output=self.path.__str__(),
        )

    @cached_property
    def hrnet_checkpoint(self) -> str:
        result = (
            self.path
            .joinpath('hrnetv2_w48_imagenet_pretrained.pth')
            .absolute().__fspath__()
        )
        return result

    @cached_property
    def snapshot(self) -> str:
        result = (
            self.path
            .joinpath('satellite_2021.pth')
            .absolute().__fspath__()
        )
        return result

    @cached_property
    def black(self) -> Path:
        dim: int = self.intiles.tile.dimension
        path: Path = self.path.joinpath(str(dim), 'black.png')

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (dim, dim), (0, 0, 0)).save(path)

        return path


static = Static()
