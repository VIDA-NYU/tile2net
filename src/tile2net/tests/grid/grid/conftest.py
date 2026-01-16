from __future__ import annotations

import pytest

from tile2net.grid.grid import Grid


@pytest.fixture(scope="session")
def grid_boston_common():
    return Grid.from_location('Boston Common, MA', zoom=20)


@pytest.fixture(scope="session")
def grid_central_park_nyc():
    return Grid.from_location('Central Park, New York', zoom=20)


@pytest.fixture(scope="session")
def grid_golden_gate_sf():
    return Grid.from_location('Golden Gate Park, San Francisco', zoom=20)


@pytest.fixture(scope="session")
def grid_griffith_la():
    return Grid.from_location('Griffith Park, Los Angeles', zoom=20)


@pytest.fixture(scope="session")
def grid_discovery_seattle():
    return Grid.from_location('Discovery Park, Seattle, Washington', zoom=20)


@pytest.fixture(scope="session")
def grid_deering_oaks_portland():
    return Grid.from_location('Deering Oaks Park, Portland, Maine', zoom=20)


@pytest.fixture(scope="session")
def grid_liberty_state_jc():
    return Grid.from_location('Liberty State Park, Jersey City', zoom=20)


@pytest.fixture(scope="session")
def grid_stevens_hoboken():
    return Grid.from_location('Stevens Park, Hoboken, New Jersey', zoom=20)


@pytest.fixture(scope="session")
def grid_spring_hill_tn():
    return Grid.from_location('Spring Hill, Tennessee', zoom=20)


@pytest.fixture(scope="session")
def grid_capital_square_richmond():
    return Grid.from_location('Capital Square, Richmond, Virginia', zoom=20)


@pytest.fixture(scope="session")
def grid_tilden_berkeley():
    return Grid.from_location('Tilden Park, Berkeley, California', zoom=20)


@pytest.fixture(scope="session")
def grid_central_park_fremont():
    return Grid.from_location('Central Park, Fremont, California', zoom=20)


@pytest.fixture(scope="session")
def grid_lake_merritt_oakland():
    return Grid.from_location('Lake Merritt, Oakland, California', zoom=20)


@pytest.fixture(
    scope="session",
    params=[
        "grid_boston_common",
        "grid_central_park_nyc",
        "grid_golden_gate_sf",
        "grid_griffith_la",
        "grid_discovery_seattle",
        "grid_deering_oaks_portland",
        "grid_liberty_state_jc",
        "grid_stevens_hoboken",
        "grid_spring_hill_tn",
        "grid_capital_square_richmond",
        "grid_tilden_berkeley",
        "grid_central_park_fremont",
        "grid_lake_merritt_oakland",
    ],
)
def grid(request):
    return request.getfixturevalue(request.param)
