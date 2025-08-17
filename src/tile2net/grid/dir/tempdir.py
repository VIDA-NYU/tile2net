from __future__ import annotations
from .ingrid import InGrid
from .vecgrid import VecGrid
from .seggrid import SegGrid


import os
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import os

import datetime
import os.path

import pandas as pd
from pandas.tseries.holiday import USPresidentsDay

from .batchiterator import BatchIterator
from .dir import Dir, Dir, Dir, Dir
from .outdir import Outdir

if False:
    import tile2net.grid.ingrid


class TempDir(Outdir):
    ...

