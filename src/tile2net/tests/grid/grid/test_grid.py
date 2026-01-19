from __future__ import annotations

from tile2net.grid import Grid
from tile2net.tests.grid.grid.conftest import fixtures


class TestGridInstantiation:
    def test_grid_is_instance(self, grid: Grid):
        assert isinstance(grid, Grid)


if __name__ == '__main__':
    for fixture in fixtures():
        print(f"\nTesting {fixture.__name__}")
        grid = fixture.__wrapped__()
        TestGridInstantiation().test_grid_is_instance(grid)
        print(f"  ✓ Grid instantiation test passed")
