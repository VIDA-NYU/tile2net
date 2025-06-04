from __future__ import annotations

from functools import *
from pathlib import Path
from typing import *

import gdown

if False:
    from .tiles import Tiles


class Static:
    instance: Tiles
    owner: type[Tiles]

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    def __get__(
            self,
            instance: Tiles,
            owner: type[Tiles]
    ):
        self.instance = instance
        self.owner = owner
        return self

    def __call__(
            self,
            *args
    ) -> Static:
        ...

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


static = Static()
