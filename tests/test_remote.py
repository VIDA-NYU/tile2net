import os.path
import shutil

import pytest
from tile2net.raster.raster import Raster
import tile2net.raster.source
from tile2net.raster.source import Source
import abc

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

def test_sources():
    for key in dir(tile2net.raster.source):
        cls = getattr(tile2net.raster.source, key)
        if (
            not isinstance(cls, type)
            or not issubclass(cls, Source)
            or abc.ABC in cls.__bases__
        ):
            continue
        # assert querying by the polygon returns the same source
        assert Source[cls.coverage.unary_union] == cls
        # assert querying by the name returns the same source
        assert Source[cls.name] == cls



if __name__ == '__main__':
    # test_small()
    test_sources()
