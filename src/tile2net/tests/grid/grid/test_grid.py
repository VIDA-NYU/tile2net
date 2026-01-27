from __future__ import annotations

import pytest

from tile2net.grid import Grid
from tile2net.tests.grid.grid.locations import LOCATIONS


class TestGridInstantiation:
    @pytest.mark.parametrize(
        "coords,zoom",
        LOCATIONS.values(),
        ids=LOCATIONS.keys()
    )
    def test_grid_is_instance(self, coords, zoom):
        grid = Grid.from_location(coords, zoom=zoom)
        assert isinstance(grid, Grid)


class TestGridAttributes:
    @pytest.mark.parametrize(
        "coords,zoom",
        LOCATIONS.values(),
        ids=LOCATIONS.keys()
    )
    def test_zoom(self, coords, zoom):
        """Test that Grid.from_location preserves input zoom level."""
        grid = Grid.from_location(coords, zoom=zoom)
        assert grid.zoom == zoom


if __name__ == '__main__':
    test_instantiation = TestGridInstantiation()
    test_attrs = TestGridAttributes()
    for name, (coords, zoom) in LOCATIONS.items():
        print(f"\nTesting {name}")

        test_instantiation.test_grid_is_instance(coords, zoom)
        print(f"  ✓ Grid instantiation test passed")

        test_attrs.test_zoom(coords, zoom)
        print(f"  ✓ zoom preservation test passed")
