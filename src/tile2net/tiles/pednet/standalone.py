from __future__ import annotations

from functools import *
from typing import *

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.ops
from geopandas import GeoDataFrame
from pandas import Series

from ..fixed import GeoDataFrameFixed
import pandas as pd
from .lines import  Lines
from .nodes import Nodes
from .edges import  Edges

if False:
    from .pednet import PedNet
    import folium
    from .stubs import Stubs
    from .mintrees import Mintrees

"""
without magicpandas,
input: lines
output: nodes, aggregated

one to drop degree=2 noeds
one to extract node information
"""





if __name__ == '__main__':
    import geopandas as gpd

    file = '/home/arstneio/PycharmProjects/kashi/src/tile2net/artifacts/static/brooklyn.feather'
    result = (
        gpd.read_feather(file)
        .pipe(Lines.from_frame)
        .drop2nodes
    )
    print(f'{len(result)=}')
    print(f'{len(result)=}')
    result.explore()
