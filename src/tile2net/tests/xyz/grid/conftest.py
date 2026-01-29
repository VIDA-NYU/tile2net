from __future__ import annotations

import pytest

from tile2net.xyz.grid import Grid
from tile2net.tests.xyz.grid.locations import LOCATIONS


@pytest.fixture(
    scope="session",
    params=LOCATIONS.values(),
    ids=LOCATIONS.keys(),
)
def grid(request) -> Grid:
    coords, zoom = request.param
    return Grid.from_location(
        coords,
        zoom=zoom,
    )


def iter_grids():
    """
    Yields (name, Grid) tuples for all configured locations.
    Useful for maintenance scripts or manual debugging.
    """
    for name, (coords, zoom) in LOCATIONS.items():
        yield name, Grid.from_location(
            coords,
            zoom=zoom,
        )
