from __future__ import annotations, absolute_import, division

import os
import sys
from pathlib import Path
from typing import Optional

from .inference import Inference

if False:
    pass
import hashlib


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
AutoResume = None

if False:
    from .tiles import Tiles


class Infer:
    tiles: Tiles

    def __get__(
            self,
            instance: Tiles,
            owner: type[Tiles],
    ):
        self.tiles = instance
        self.Tiles = owner
        return self

    def __call__(
            self,
    ):
        tiles = self.tiles
        args = self.tiles.cfg
        inference = Inference(
            tiles,
        )

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...
