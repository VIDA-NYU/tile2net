import shutil

import pytest
from tile2net.raster.raster import Raster

def test_small():
    raster = Raster(
        location='Washington Square Park, New York, NY, USA',
        zoom=19,
        # dump_percent=1,
        # name='small'
        name='tiny'
    )

    raster.generate(2)
    raster.inference('--local', '--debug')

if __name__ == '__main__':
    test_small()
