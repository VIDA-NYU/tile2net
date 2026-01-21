from __future__ import annotations

from tile2net.tests.grid.grid.conftest import *


class TestGridInstantiation:
    def test_grid_is_instance(self, grid: Grid):
        assert isinstance(grid, Grid)


if __name__ == '__main__':
    for name, grid in iter_grids():
        print(f"\nTesting {name}")
        TestGridInstantiation().test_grid_is_instance(grid)
        print(f"  ✓ Grid instantiation test passed")
