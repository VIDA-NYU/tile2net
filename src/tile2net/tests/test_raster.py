import shutil

import pytest
from tile2net.raster.raster import Raster

class TestRaster:


    def test_fail(self):
        raise Exception('test failed')


    # def test_washington(self):
    #     raster = Raster(
    #         location='Washington Square Park, New York, NY, USA',
    #         zoom=18,
    #         dump_percent=10,
    #     )
    #     raster.generate(2)
    #     raster.inference()
    #     shutil.rmtree(raster.project)
    #
    #

