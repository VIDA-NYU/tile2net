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


class Infer:
    tiles: Tiles


    def to_dimension(
            self,
            dimension: int = 1024,
            pad: bool = True,
    ) -> Tiles:
        """"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        ...



