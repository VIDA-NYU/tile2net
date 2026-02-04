from __future__ import annotations

import pytest

from tile2net.geo.ingrid import InGrid
from tile2net.tests.geo.ingrid.locations import LOCATIONS


@pytest.fixture(
    scope="session",
    params=LOCATIONS.values(),
    ids=LOCATIONS.keys(),
)
def grid(request) -> InGrid:
    coords, zoom = request.param
    return InGrid.from_location(
        coords,
        zoom=zoom,
    )


def iter_grids():
    """
    Yields (name, InGrid) tuples for all configured locations.
    Useful for maintenance scripts or manual debugging.
    """
    for name, (coords, zoom) in LOCATIONS.items():
        yield name, InGrid.from_location(
            coords,
            zoom=zoom,
        )
