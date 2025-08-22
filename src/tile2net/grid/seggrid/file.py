from __future__ import annotations
from ...grid.frame.namespace import namespace

import copy
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from functools import *
from typing import *

import PIL.Image
import imageio.v3 as iio
import numpy
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tile2net.grid.cfg.cfg import assert_and_infer_cfg
from tile2net.grid.cfg.logger import logger
from tile2net.grid.grid.static import Static
from tile2net.grid.tileseg import datasets
from tile2net.grid.tileseg import network
from tile2net.grid.tileseg.loss.optimizer import get_optimizer, restore_opt, restore_net
from tile2net.grid.tileseg.loss.utils import get_loss
from tile2net.grid.tileseg.network.ocrnet import MscaleOCR
from tile2net.grid.tileseg.utils.misc import AverageMeter, prep_experiment
from . import delayed
from .minibatch import MiniBatch
from .vectile import VecTile
from .. import frame
from ..grid import file
from ..grid.grid import Grid
from ..util import recursion_block
from .. import util
from .padded import Padded

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))

if False:
    from .filled import Filled
    from .broadcast import Broadcast
    from ..ingrid import InGrid
    from .seggrid import SegGrid


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


class File(
    file.File
):
    grid: SegGrid

    @frame.column
    def infile(self) -> pd.Series:
        """
        A file for each segmentation tile: the stitched input grid.
        Stitches input files when seggrid.file is accessed
        """
        grid = self.grid
        files = grid.ingrid.tempdir.seggrid.infile.files(grid)
        self.infile = files
        if not files.map(os.path.exists).all():
            ingrid = grid.ingrid
            _ = ingrid.file.infile
            big_files = ingrid.segtile.infile
            assert (
                ingrid.file.infile
                .map(os.path.exists)
                .all()
            )
            ingrid._stitch(
                small_grid=ingrid,
                big_grid=grid,
                r=ingrid.segtile.r,
                c=ingrid.segtile.c,
                small_files=ingrid.file.infile,
                big_files=big_files,
            )
            msg = f"Files not stitched: {files[~files.map(os.path.exists)]}"
            assert files.map(os.path.exists).all(), msg

        return files

    @frame.column
    def grayscale(self) -> pd.Series:
        """Segmentation masks, where each pixel is a class id"""
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.grayscale.files(grid)
        self.grayscale = files
        if not files.map(os.path.exists).all():
            grid.predict()
        return files

    @frame.column
    def probability(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.prob.files(grid)
        self.probability = files
        if not files.map(os.path.exists).all():
            grid.predict()
        return files

    @frame.column
    def error(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.error.files(grid)
        self.error = files
        if not files.map(os.path.exists).all():
            grid.predict()
        return files

    @frame.column
    def submit(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.submit.files(grid)
        self.submit = files
        if not files.map(os.path.exists).all():
            grid.predict()
        return files

    @frame.column
    def colored(self) -> pd.Series:
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.colored.files(grid)
        self.colored = files
        if not files.map(os.path.exists).all():
            grid.predict()
        return files

    def output(self, dirname: str) -> pd.Series:
        # Note: Can't use frame.column with parameters
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.output.files(grid, dirname)
        key = f'output.{dirname}'
        setattr(self, key, files)
        if not files.map(os.path.exists).all():
            grid.predict()
        return files
