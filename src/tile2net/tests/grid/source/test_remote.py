from __future__ import annotations

from io import BytesIO

import pytest
import requests
from PIL import Image

from tile2net.grid import util
from tile2net.grid.source.remote import Remote

PROTOTYPES = {
    key: value
    for key, value in Remote.name2prototype.items()
}


@pytest.fixture(
    scope="session",
    ids=PROTOTYPES.keys(),
    params=PROTOTYPES.values()
)
def source(source) -> Remote:
    return source


class TestPrototypes:

    def test_get_urls(self, prototype: Remote):
        """
        Test that a URL generated for the prototype's coverage yields a valid image
        matching the expected dimensions.
        """
        if prototype.coverage.empty:
            pytest.skip(f"No coverage found for {prototype}")

        geometry = prototype.coverage.geometry.iloc[0]
        centroid = geometry.centroid
        zoom = prototype.zoom

        x, y = util.lonlat2xy(centroid.x, centroid.y, zoom)

        # note: in case of debugging, uncomment and change this
        # zoom = 19
        # y, x = 37.52062708311975, -77.43985611243849
        # x, y = util.lonlat2xy(x, y, zoom)

        try:
            url = (
                prototype
                .get_urls([x], [y], zoom)
                .__next__()
            )
        except StopIteration:
            pytest.fail(f"get_urls yielded no URLs for {prototype} at {x=}, {y=}, {zoom=}")

        # maintain a User-Agent to avoid blocking by some tile servers
        headers = {'User-Agent': 'tile2net/test'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            pytest.fail(f"Failed to fetch tile from {url}: {e}")

        # verify image dimensions
        try:
            content = BytesIO(response.content)
            image = Image.open(content)
            msg = (
                f"Image dimensions {image.size} do not match "
                f"prototype dimension {prototype.dimension}"
            )
            assert image.size == (prototype.dimension, prototype.dimension), msg
        except IOError:
            pytest.fail(f"Returned content from {url} is not a valid image.")


    def test_locations(self, prototype: Remote):
        # todo: include locations under the `test` parameter for each prototype in `servers.yaml`;
        #   each prototype will iterate across these locations and assure that Remote.from_location() returns a copy
        #   of that prototype
        """"""



if __name__ == '__main__':
    test = TestPrototypes()
    for name, prototype in PROTOTYPES.items():
        print(f'  Testing get_urls for prototype: {name}')
        test.test_get_urls(prototype)
        print("  ✓ get_urls works correctly")
