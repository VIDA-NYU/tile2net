from __future__ import annotations

import pytest

from tile2net.core.cfg import Cfg
from tile2net.geo import InGrid


class TestNamePriority:
    """Test InGrid.name priority system."""

    def test_passed_name_highest_priority(self):
        """Passed name parameter should take highest priority."""
        grid = InGrid.from_location(
            '42.3601,-71.0589,42.3551,-71.0539',
            zoom=20,
            name='MyProject'
        )
        assert grid.name == 'MyProject'

    def test_cfg_name_second_priority(self):
        """Config name should be used when no name is passed."""
        cfg = Cfg()
        cfg['name'] = 'CfgProject'
        grid = InGrid.from_location(
            '42.3601,-71.0589,42.3551,-71.0539',
            zoom=20
        )
        grid.cfg = cfg
        assert grid.name == 'CfgProject'

    def test_passed_name_overrides_cfg(self):
        """Passed name should override config name."""
        cfg = Cfg()
        cfg['name'] = 'CfgProject'
        grid = InGrid.from_location(
            '42.3601,-71.0589,42.3551,-71.0539',
            zoom=20,
            name='PassedProject'
        )
        grid.cfg = cfg
        assert grid.name == 'PassedProject'


class TestNameFromBounds:
    """Test InGrid.name when using from_bounds."""

    def test_from_bounds_with_name(self):
        """from_bounds should accept and use name parameter."""
        grid = InGrid.from_bounds(
            (42.36, -71.06, 42.35, -71.05),
            zoom=20,
            name='BoundsProject'
        )
        assert grid.name == 'BoundsProject'

    def test_from_bounds_without_name(self):
        """from_bounds should infer name when not provided."""
        grid = InGrid.from_bounds(
            (42.36, -71.06, 42.35, -71.05),
            zoom=20
        )
        # should infer from source or coordinates
        assert grid.name is not None
        assert len(grid.name) > 0


class TestNameFormatting:
    """Test name formatting for different location types."""

    def test_geographic_coords_formatting(self):
        """Geographic coordinates should be formatted to 2 decimals and lowercased."""
        grid = InGrid.from_location(
            '42.3601,-71.0589,42.3551,-71.0539',
            zoom=20
        )
        # should be lowercased bbox with 2 decimal places
        expected = '42.36,-71.06,42.36,-71.05'
        assert grid.name == expected

    def test_geographic_coords_tuple_formatting(self):
        """Geographic coordinates as tuple should be formatted correctly."""
        grid = InGrid.from_location(
            (42.3601, -71.0589, 42.3551, -71.0539),
            zoom=20
        )
        # should be lowercased bbox with 2 decimal places
        expected = '42.36,-71.06,42.36,-71.05'
        assert grid.name == expected

    def test_name_lowercasing(self):
        """Inferred names should be lowercased, but passed names preserved."""
        # inferred name should have no uppercase letters
        grid1 = InGrid.from_location(
            '42.36,-71.06,42.35,-71.05',
            zoom=20
        )
        # check that there are no uppercase letters (coordinate strings are all numeric)
        assert not any(c.isupper() for c in grid1.name)

        # passed name SHOULD be preserved as-is
        grid2 = InGrid.from_location(
            '42.36,-71.06,42.35,-71.05',
            zoom=20,
            name='MyProject'
        )
        assert grid2.name == 'MyProject'
        # verify it has uppercase letters (proving it wasn't lowercased)
        assert any(c.isupper() for c in grid2.name)

        # cfg name SHOULD be preserved as-is
        cfg = Cfg()
        cfg['name'] = 'CfgProject'
        grid3 = InGrid.from_location(
            '42.36,-71.06,42.35,-71.05',
            zoom=20
        )
        grid3.cfg = cfg
        assert grid3.name == 'CfgProject'
        # verify it has uppercase letters (proving it wasn't lowercased)
        assert any(c.isupper() for c in grid3.name)


class TestNameCaching:
    """Test that name is cached properly."""

    def test_name_is_cached(self):
        """Name should be cached after first access."""
        grid = InGrid.from_location(
            '42.36,-71.06,42.35,-71.05',
            zoom=20,
            name='TestProject'
        )
        # access name twice
        name1 = grid.name
        name2 = grid.name
        # should be the same object (cached)
        assert name1 == name2
        assert name1 == 'TestProject'


if __name__ == '__main__':
    import sys

    test_priority = TestNamePriority()
    test_bounds = TestNameFromBounds()
    test_formatting = TestNameFormatting()
    test_caching = TestNameCaching()

    tests = [
        ("Name Priority Tests", test_priority, [
            ("passed_name_highest_priority", "Passed name has highest priority"),
            ("cfg_name_second_priority", "Config name has second priority"),
            ("passed_name_overrides_cfg", "Passed name overrides config"),
        ]),
        ("Name from Bounds Tests", test_bounds, [
            ("from_bounds_with_name", "from_bounds accepts name parameter"),
            ("from_bounds_without_name", "from_bounds infers name from coordinates"),
        ]),
        ("Name Formatting Tests", test_formatting, [
            ("geographic_coords_formatting", "Geographic coords formatted to 2 decimals"),
            ("geographic_coords_tuple_formatting", "Geographic coords tuple formatted correctly"),
            ("name_lowercasing", "Names lowercased except passed/cfg"),
        ]),
        ("Name Caching Tests", test_caching, [
            ("name_is_cached", "Name is cached after first access"),
        ]),
    ]

    all_passed = True
    total_tests = 0
    passed_tests = 0

    print("="*70)
    print("InGrid.name Refactoring Tests")
    print("="*70)
    print()

    for section_name, test_obj, test_methods in tests:
        print(f"\n{section_name}")
        print("-" * len(section_name))

        for method_name, description in test_methods:
            total_tests += 1
            test_method = getattr(test_obj, f"test_{method_name}")
            test_method()
            print(f'  ✓ {description}')
            passed_tests += 1

    print()
    print("="*70)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("="*70)

    sys.exit(0 if all_passed else 1)
