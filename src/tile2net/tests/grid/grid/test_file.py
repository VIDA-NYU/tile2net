from __future__ import annotations

import os

from tile2net.tests.grid.grid.conftest import *


class TestFileStatic:
    def test_static_exists(self, grid: Grid):
        assert (
            grid.file.static
            .map(os.path.exists)
            .all()
        )


class TestFilePred:
    def test_pred_exists(self, grid: Grid):
        assert (
            grid.file.pred
            .map(os.path.exists)
            .all()
        )


class TestFileProb:
    def test_prob_exists(self, grid: Grid):
        assert (
            grid.file.prob
            .map(os.path.exists)
            .all()
        )


class TestFileNetwork:
    def test_network_exists(self, grid: Grid):
        assert os.path.exists(grid.file.network)


class TestFilePolygons:
    def test_polygons_exists(self, grid: Grid):
        assert os.path.exists(grid.file.polygons)


if __name__ == '__main__':

    for name, grid in iter_grids():
        print(f"\nTesting {name}")
        TestFileStatic().test_static_exists(grid)
        TestFilePred().test_pred_exists(grid)
        TestFileProb().test_prob_exists(grid)
        TestFileNetwork().test_network_exists(grid)
        TestFilePolygons().test_polygons_exists(grid)
