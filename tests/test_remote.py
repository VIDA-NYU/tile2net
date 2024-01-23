import shutil

import pytest
from tile2net.raster.raster import Raster

def test_small():
    raster = Raster(
        location='Washington Square Park, New York, NY, USA',
        zoom=19,
        dump_percent=10,
        name='small'
    )
    raster.generate(2)
    raster.inference('--remote')

if __name__ == '__main__':
    test_small()
