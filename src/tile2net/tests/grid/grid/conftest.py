from __future__ import annotations

import pytest

from tile2net.grid.grid import Grid


@pytest.fixture(scope="session")
def grid_boston_common():
    # return Grid.from_location('Boston Common, MA')
    return Grid.from_location((317280, 387840, 317312, 387872), zoom=20)


@pytest.fixture(scope="session")
def grid_central_park_nyc():
    # return Grid.from_location('Central Park, New York')
    return Grid.from_location((308800, 393888, 308896, 394048), zoom=20)


@pytest.fixture(scope="session")
def grid_golden_gate_sf():
    # return Grid.from_location('Golden Gate Park, San Francisco')
    return Grid.from_location((167440, 405296, 167632, 405344), zoom=20)


@pytest.fixture(scope="session")
def grid_griffith_la():
    # return Grid.from_location('Griffith Park, Los Angeles')
    return Grid.from_location((179584, 418304, 179808, 418496), zoom=20)


@pytest.fixture(scope="session")
def grid_discovery_seattle():
    # return Grid.from_location('Discovery Park, Seattle, Washington')
    return Grid.from_location((167648, 365920, 167776, 366016), zoom=20)


@pytest.fixture(scope="session")
def grid_deering_oaks_portland():
    # return Grid.from_location('Deering Oaks Park, Portland, Maine')
    return Grid.from_location((159792, 191312, 159824, 191344), zoom=19)


@pytest.fixture(scope="session")
def grid_liberty_state_jc():
    # return Grid.from_location('Liberty State Park, Jersey City')
    return Grid.from_location((308544, 394240, 308672, 394336), zoom=20)


@pytest.fixture(scope="session")
def grid_stevens_hoboken():
    # return Grid.from_location('Stevens Park, Hoboken, New Jersey')
    return Grid.from_location((308664, 394128, 308672, 394144), zoom=20)


@pytest.fixture(scope="session")
def grid_spring_hill_tn():
    # return Grid.from_location('Spring Hill, Tennessee')
    return Grid.from_location((270880, 412480, 271328, 412896), zoom=20)


@pytest.fixture(scope="session")
def grid_capital_square_richmond():
    # return Grid.from_location('Capital Square, Richmond, Virginia')
    return Grid.from_location((298632, 406056, 298640, 406072), zoom=20)


@pytest.fixture(scope="session")
def grid_tilden_berkeley():
    # return Grid.from_location('Tilden Park, Berkeley, California')
    return Grid.from_location((168128, 404768, 168320, 404928), zoom=20)


@pytest.fixture(scope="session")
def grid_central_park_fremont():
    # return Grid.from_location('Central Park, Fremont, California')
    return Grid.from_location((169056, 406144, 169088, 406208), zoom=20)


@pytest.fixture(scope="session")
def grid_lake_merritt_oakland():
    # return Grid.from_location('Lake Merritt, Oakland, California')
    return Grid.from_location((168160, 405152, 168224, 405216), zoom=20)


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


def fixtures():
    yield grid_boston_common
    yield grid_central_park_nyc
    yield grid_golden_gate_sf
    yield grid_griffith_la
    yield grid_discovery_seattle
    yield grid_deering_oaks_portland
    yield grid_liberty_state_jc
    yield grid_stevens_hoboken
    yield grid_spring_hill_tn
    yield grid_capital_square_richmond
    yield grid_tilden_berkeley
    yield grid_central_park_fremont
    yield grid_lake_merritt_oakland
