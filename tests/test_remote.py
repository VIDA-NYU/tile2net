import os.path
import shutil

import pytest
from tile2net.raster.raster import Raster

def test_small():
    raster = Raster(
        location='Washington Square Park, New York, NY, USA',
        zoom=19,
        # dump_percent=10,
        name='small'
    )
    # for file in raster.project.resources.segmentation.files():
        # assert file.exists()
        # assert os.path.exists(file)
    raster.generate(2)
    raster.inference('--remote', '--debug')


if __name__ == '__main__':
    test_small()

