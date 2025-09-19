from __future__ import annotations
import ctypes

from concurrent.futures import ThreadPoolExecutor, Future, wait
from threading import BoundedSemaphore
from pathlib import Path
import os, tempfile, uuid

import copy
import gc
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
from functools import *
from pathlib import Path
from typing import *
from typing import TYPE_CHECKING

import cv2
import numpy
import torch
import torch.distributed as dist
from torch.nn.parallel.data_parallel import DataParallel
from tqdm import tqdm
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd
from functools import *
from typing import *
from types import *
from shapely import *
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from itertools import *
from pandas.api.extensions import ExtensionArray


class Submitter:
    def __init__(
            self,
            workers: int = 4,
            max_inflight: int = 8,
    ):
        self.workers = workers
        self.max_inflight = max_inflight

    @cached_property
    def gate(self):
        return BoundedSemaphore()

    @cached_property
    def threads(self):
        return ThreadPoolExecutor(max_workers=self.workers)

    @cached_property
    def futures(self):
        return []

    @cached_property
    def errors(self):
        return []

    def _imwrite(
            self,
            filename: str,
            *args,
            **kwargs
    ):
        try:
            p = Path(filename)
            fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(p.parent))
            os.close(fd)
            try:
                cv2.imwrite(filename, *args, **kwargs)
                os.replace(tmp, p)
            except Exception:
                try:
                    os.unlink(tmp)
                finally:
                    raise
        finally:
            self.gate.release()

    def imwrite(
            self,
            filename: str,
            img,
            params,
    ):
        self.gate.acquire()
        try:
            return self.threads.submit(self, self._imwrite, filename, img, params)
        except Exception:
            self.gate.release()
            raise

    def drain(self):
        done, _ = wait(self.futures)
        errors = self.errors
        for fut in done:
            try:
                exc = fut.exception()
            except Exception:
                exc = None
            if exc is not None:
                errors.append(exc)
        self.futures.clear()

    def rotate(self):
        self.threads.shutdown(wait=True)
        # malloc trim if supported (posix only)
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
        del self.threads


