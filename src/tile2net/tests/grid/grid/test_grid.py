from __future__ import annotations

from tile2net.grid.grid import Grid


class TestFromAddress:
    def test_boston_common(self):
        grid = Grid.from_location('Boston Common, MA', )
        assert isinstance(grid, Grid)

    def test_central_park_nyc(self):
        grid = Grid.from_location('Central Park, New York', )
        assert isinstance(grid, Grid)

    def test_golden_gate_sf(self):
        grid = Grid.from_location('Golden Gate Park, San Francisco', )
        assert isinstance(grid, Grid)

    def test_griffith_la(self):
        grid = Grid.from_location('Griffith Park, Los Angeles', )
        assert isinstance(grid, Grid)

    def test_discovery_seattle(self):
        grid = Grid.from_location('Discovery Park, Seattle, Washington', )
        assert isinstance(grid, Grid)

    def test_deering_oaks_portland(self):
        grid = Grid.from_location('Deering Oaks Park, Portland, Maine', )
        assert isinstance(grid, Grid)

    def test_liberty_state_jc(self):
        grid = Grid.from_location('Liberty State Park, Jersey City', )
        assert isinstance(grid, Grid)

    def test_stevens_hoboken(self):
        grid = Grid.from_location('Stevens Park, Hoboken, New Jersey', )
        assert isinstance(grid, Grid)

    def test_spring_hill_tn(self):
        grid = Grid.from_location('Spring Hill, Tennessee', )
        assert isinstance(grid, Grid)

    def test_capital_square_richmond(self):
        grid = Grid.from_location('Capital Square, Richmond, Virginia', )
        assert isinstance(grid, Grid)

    def test_tilden_berkeley(self):
        grid = Grid.from_location('Tilden Park, Berkeley, California', )
        assert isinstance(grid, Grid)

    def test_central_park_fremont(self):
        grid = Grid.from_location('Central Park, Fremont, California', )
        assert isinstance(grid, Grid)

    def test_lake_merritt_oakland(self):
        grid = Grid.from_location('Lake Merritt, Oakland, California', )
        assert isinstance(grid, Grid)


if __name__ == '__main__':
    test = TestFromAddress()
    test_methods = [
        'test_boston_common',
        'test_central_park_nyc',
        'test_golden_gate_sf',
        'test_griffith_la',
        'test_discovery_seattle',
        'test_deering_oaks_portland',
        'test_liberty_state_jc',
        'test_stevens_hoboken',
        'test_spring_hill_tn',
        'test_capital_square_richmond',
        'test_tilden_berkeley',
        'test_central_park_fremont',
        'test_lake_merritt_oakland',
    ]

    for method_name in test_methods:
        print(f"\nTesting {method_name}")
        getattr(test, method_name)()
        print(f"  ✓ {method_name} passed")
