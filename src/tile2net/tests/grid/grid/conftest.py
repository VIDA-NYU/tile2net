from __future__ import annotations

import pytest

from tile2net.grid.grid import Grid

LOCATIONS = dict(
    boston_common=(
        (317280, 387840, 317312, 387872),
        20
    ),
    central_park_nyc=(
        (308800, 393888, 308896, 394048),
        20
    ),
    golden_gate_sf=(
        (167440, 405296, 167632, 405344),
        20
    ),
    griffith_la=(
        (179584, 418304, 179808, 418496),
        20
    ),
    discovery_seattle=(
        (167648, 365920, 167776, 366016),
        20
    ),
    deering_oaks_portland=(
        (159792, 191312, 159824, 191344),
        19
    ),
    liberty_state_jc=(
        (308544, 394240, 308672, 394336),
        20
    ),
    stevens_hoboken=(
        (308664, 394128, 308672, 394144),
        20
    ),
    spring_hill_tn=(
        (270880, 412480, 271328, 412896),
        20
    ),
    capital_square_richmond=(
        (298632, 406056, 298640, 406072),
        20
    ),
    tilden_berkeley=(
        (168128, 404768, 168320, 404928),
        20
    ),
    central_park_fremont=(
        (169056, 406144, 169088, 406208),
        20
    ),
    lake_merritt_oakland=(
        (168160, 405152, 168224, 405216),
        20
    ),
)


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
