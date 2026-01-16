from __future__ import annotations

import pytest

from tile2net.grid.grid import Grid


@pytest.fixture(scope="session")
def grid_boston():
    return Grid.from_location('Boston Common, MA', zoom=20)


@pytest.fixture(scope="session")
def grid_nyc():
    return Grid.from_location('New York City', zoom=20)


@pytest.fixture(scope="session")
def grid_nyc_alias():
    return Grid.from_location('City of New York', zoom=20)


@pytest.fixture(scope="session")
def grid_mass():
    return Grid.from_location('Massachusetts', zoom=20)


@pytest.fixture(scope="session")
def grid_king_county():
    return Grid.from_location('King County, Washington', zoom=20)


@pytest.fixture(scope="session")
def grid_king_alias():
    return Grid.from_location('King County', zoom=20)


@pytest.fixture(scope="session")
def grid_la():
    return Grid.from_location('Los Angeles', zoom=20)


@pytest.fixture(scope="session")
def grid_nj():
    return Grid.from_location('New Jersey', zoom=20)


@pytest.fixture(scope="session")
def grid_nj_portland():
    return Grid.from_location('Portland, Maine', zoom=20)


@pytest.fixture(scope="session")
def grid_nj_brunswick():
    return Grid.from_location('New Brunswick, New Jersey', zoom=20)


@pytest.fixture(scope="session")
def grid_nj_jc():
    return Grid.from_location('Jersey City', zoom=20)


@pytest.fixture(scope="session")
def grid_nj_hoboken():
    return Grid.from_location('Hoboken', zoom=20)


@pytest.fixture(scope="session")
def grid_spring_hill():
    return Grid.from_location('Spring Hill, Tennessee', zoom=20)


@pytest.fixture(scope="session")
def grid_va():
    return Grid.from_location('Virginia', zoom=20)


@pytest.fixture(scope="session")
def grid_me():
    return Grid.from_location('Maine', zoom=20)


@pytest.fixture(scope="session")
def grid_berkeley():
    return Grid.from_location('Berkeley, California', zoom=20)


@pytest.fixture(scope="session")
def grid_fremont():
    return Grid.from_location('Fremont, California', zoom=20)


@pytest.fixture(scope="session")
def grid_oakland():
    return Grid.from_location('Oakland, California', zoom=20)


@pytest.fixture(scope="session")
def grid_sf():
    return Grid.from_location('San Francisco, California', zoom=20)


@pytest.fixture(
    scope="session",
    params=[
        "grid_boston",
        "grid_nyc",
        "grid_nyc_alias",
        "grid_mass",
        "grid_king_county",
        "grid_king_alias",
        "grid_la",
        "grid_nj",
        "grid_nj_portland",
        "grid_nj_brunswick",
        "grid_nj_jc",
        "grid_nj_hoboken",
        "grid_spring_hill",
        "grid_va",
        "grid_me",
        "grid_berkeley",
        "grid_fremont",
        "grid_oakland",
        "grid_sf",
    ],
)
def grid(request):
    return request.getfixturevalue(request.param)
