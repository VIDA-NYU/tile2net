from __future__ import annotations

import os
from tile2net.grid.grid import Grid


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
