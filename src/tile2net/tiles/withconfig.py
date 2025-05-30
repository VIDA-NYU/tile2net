from __future__ import annotations
from tqdm import tqdm

import os.path

from pathlib import Path
from sympy.core.benchmarks.bench_assumptions import timeit_x_is_integer
from tempfile import gettempdir
from uuid import uuid4

import tempfile

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import imageio.v2

import math
import numpy as np
import pandas as pd
from typing import *
from .indir import Loader
from tile2net.logger import logger

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
