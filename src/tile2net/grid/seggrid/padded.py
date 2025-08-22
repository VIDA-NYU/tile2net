from __future__ import annotations

from .. import util

import copy
import os
import os.path
import shutil
import sys
from ..util import ensure_tempdir_for_indir
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import *
from pathlib import Path
from typing import *

import certifi
import geopandas as gpd
import imageio.v3 as iio
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from tile2net.grid.cfg.logger import logger
from tile2net.grid.dir.indir import Indir
from tile2net.grid.dir.outdir import Outdir
from .. import frame

from ...grid.frame.namespace import namespace

if False:
    from .seggrid import SegGrid


class Padded(
    namespace
):
    instance: SegGrid

    @property
    def length(self) -> int:
        result = self.instance.length + 2
        return result

    @property
    def dimension(self) -> int:
        return self.instance.ingrid.dimension * self.length

    @property
    def grid(self) -> SegGrid:
        return self.instance

    @frame.column
    def infile(self) -> pd.Series:
        """
        A file for each segmentation tile: the stitched input grid.
        Stitches input files when seggrid.file is accessed
        """
        seggrid = self.grid
        files = seggrid.ingrid.tempdir.seggrid.padded.infile.files(seggrid)

        self.infile = files
        if not files.map(os.path.exists).all():
            ingrid = seggrid.ingrid.broadcast
            small_files = ingrid.file.infile
            big_files = ingrid.segtile.infile
            assert (
                ingrid.file.infile
                .map(os.path.exists)
                .all()
            )
            ingrid._stitch(
                small_grid=ingrid,
                big_grid=seggrid,
                r=ingrid.segtile.r,
                c=ingrid.segtile.c,
                small_files= small_files,
                big_files= big_files,
            )
            msg = f"Files not stitched: {files[~files.map(os.path.exists)]}"
            assert files.map(os.path.exists).all(), msg

        return files
