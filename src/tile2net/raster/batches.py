from __future__ import annotations
import copy
from functools import cached_property
from typing import Iterator
from weakref import WeakKeyDictionary

import numpy as np

if False:
    from .raster import Raster, Grid


# h, w = instance.tiles.shape
# target_size = self.target
# totals = h * w  # total items in original grid
# cols_round = int(target_size / h)  # number of full-length columns needed to make suggested batch size
# n_batch_rnd = int(w / cols_round)  # number of full batches
# batch_size = cols_round * h  # number of items in each full batch
# fbatch_total = batch_size * n_batch_rnd  # total number of items with full batches
# fbatch_cols = fbatch_total / h  # number of columns needed to get full batches
# pbatch_size = totals - fbatch_total  # size of the last (partial) batch (if any)

# self.fbatch_cols: int = fbatch_cols
# self.n_batch_rnd: int = n_batch_rnd
# self.pbatch_size: int = pbatch_size

class Batches:
    target = 700
    cache: dict[Raster, Batches] = WeakKeyDictionary()

    @cached_property
    def ncolumns(self):
        # n_batch_rnd (int): number of full batches
        ...

    def __get__(self, instance: Raster, owner: type[Raster]):
        self.owner = instance
        self.Owner = owner
        if instance not in self.cache:
            self.cache[instance] = copy.copy(self)
        return self.cache[instance]

    def __iter__(self) -> Iterator[Raster]:
        raster = self.owner
        TILES = raster.tiles
        r, c = TILES.shape
        target = self.target
        STEP = np.floor_divide(target, r)
        STEP = np.maximum(STEP, 1)
        steps = np.arange(0, c, STEP)
        for step in steps:
            raster.tiles = TILES[:, step:step + STEP]
            yield raster
        raster.tiles = TILES
