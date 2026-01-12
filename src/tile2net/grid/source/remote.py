
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Union

import pandas as pd
import shapely.geometry
from geopandas import GeoDataFrame

from tile2net.grid.geocode import GeoCode
from tile2net.grid.source.catalog import Catalog
from tile2net.grid.source.coverage import Coverage, RemoteNotFound
from tile2net.grid.source.exceptions import (
    InvalidLocation,
    InvalidRemoteName,
    SourceParseError,
)
from tile2net.grid.source.prototype import Prototype
from tile2net.grid.source.source import Source
from tile2net.logger import logger


class Remote(
    Source,
    ABC
):
    """Base class for remote tile sources (HTTP/HTTPS URLs)."""

    @Catalog
    def catalog(self):
        """
        Catalog, mapping the name to the prototype instanec, for each
        enabled Remote subclass.

        Automatically set during Remote subclass initialization:
        >>> Remote.__init_subclass__
        """

    @Coverage
    def coverage(self):
        """
        Spatial coverage GeoSeries for all registered remotes.
        Can be called to find the best matching remote for a location.
        >>> Coverage.__get__
        """

    @Prototype
    def prototype(self):
        """
        Prototype instance for each class.
        Retruns
        >>> Prototype.__get__
        """

    @cached_property
    def enabled(self):
        """
        Whether this remote source is enabled for use.
        Set False to disable it from being constructed.
        """
        return True

    @cached_property
    @abstractmethod
    def name(self) -> str:
        """Short name of the remote source, e.g. `nyc`.'"""

    @cached_property
    @abstractmethod
    def ignore(self) -> str:
        """Original URL string as provided, before parsing."""

    @cached_property
    @abstractmethod
    def original(self) -> str:
        """Original URL string as provided, before parsing."""

    @cached_property
    @abstractmethod
    def format(self) -> str:
        """URL template with {x}, {y}, {z} placeholders for tile coordinates."""

    @cached_property
    @abstractmethod
    def scheme(self) -> str:
        """URL scheme (http or https)."""

    @cached_property
    @abstractmethod
    def netloc(self) -> str:
        """Network location (domain) of the remote source."""

    @cached_property
    @abstractmethod
    def path(self) -> str:
        """URL path component."""

    @cached_property
    @abstractmethod
    def extension(self) -> str:
        """File extension without the leading dot (e.g., 'png', 'jpg')."""

    @cached_property
    @abstractmethod
    def keyword(self) -> Union[str, tuple[str, ...]]:
        """
        A keyword is a required match in the reverse geocode to resolve
        discrepancies.
        """

    @cached_property
    @abstractmethod
    def dropword(self) -> Union[str, tuple[str, ...]]:
        """
        A dropword is the reverse of a keyword. If a reverse geocode
        contains this, the remote is not relevant.
        """

    @cached_property
    @abstractmethod
    def year(self) -> int:
        """Year of the data"""

    @cached_property
    def dimension(self) -> int:
        """Default dimension of the remote grid, e.g. 256 pixels."""
        return 256

    @cached_property
    def zoom(self) -> int:
        """
        Default XYZ zoom level for the remote.
        Our model performs best with a zoom of at least 19.
        """
        return 19

    @property
    def urls(self) -> pd.Series:
        """Generate URLs for all tiles in the attached grid."""
        grid = self.grid
        if grid is None:
            raise ValueError("Remote source is not attached to a grid.")

        template = self.format
        zoom = grid.zoom

        data = [
            template.format(z=zoom, y=tile.ytile, x=tile.xtile)
            for tile in grid.tiles
        ]
        return pd.Series(data, index=grid.index, dtype='str')

    @property
    def files(self) -> pd.Series:
        """Alias for urls - remote sources provide URLs instead of file paths."""
        return self.urls

    @classmethod
    def from_location(
            cls,
            item: Union[
                str,
                list[float],
                tuple[float, ...],
                shapely.geometry.base.BaseGeometry,
                GeoSeries,
                GeoDataFrame,
            ]
    ) -> Self:
        """
        Find and instantiate the most appropriate Remote for a given location.

        item:
            str: Address to geocode
            list[float] | tuple[float, ...]: Bounding box coordinates
            BaseGeometry: Shapely geometry
            GeoSeries/GeoDataFrame: Geographic data
        """
        # Get all coverage geometries
        matches: GeoSeries = cls.coverage

        # Convert item to geocode
        if isinstance(item, (GeoSeries, GeoDataFrame)):
            infer = item.geometry.iat[0].centroid
        else:
            infer = item

        try:
            geocode = GeoCode.from_inferred(infer)
            _ = geocode.polygon
        except Exception as exc:
            msg = f"Could not geocode location: {item} \n\t{exc}"
            raise InvalidLocation(msg) from exc

        # Find intersecting coverages
        loc = matches.intersects(geocode.polygon)
        if (
                not loc.any()
                and 'address' in geocode.__dict__
        ):
            # User may have been lazy; recompute polygon
            del geocode.address
            _ = geocode.address
            del geocode.nwse
            del geocode.wsen
            del geocode.polygon
            loc = matches.intersects(geocode.polygon)
            if not loc.any():
                msg = (
                    f'No remote coverage found for location: "{item}" \n\t'
                    f'Geocoded as: "{geocode.address}" \n\t'
                    f'Please be more specific.'
                )
                raise RemoteNotFound(msg)
        matches = matches.loc[loc]

        # Resolve discrepancies using keywords
        loc = []
        for name in matches.index:
            remote_class = cls.catalog[name]
            keyword = remote_class.prototype.keyword
            dropword = remote_class.prototype.dropword

            # Check dropwords first - if present, exclude this remote
            if dropword:
                if isinstance(dropword, str):
                    dropwords = dropword,
                else:
                    dropwords = dropword

                if any(word.casefold() in geocode.address.casefold()
                       for word in dropwords):
                    loc.append(False)
                    continue

            # Check keywords
            if keyword:
                if isinstance(keyword, str):
                    loc.append(keyword.casefold() in geocode.address.casefold())
                else:
                    append = any(
                        word.casefold() in geocode.address.casefold()
                        for word in keyword
                    )
                    loc.append(append)
            else:
                # No keyword requirement, include by default
                loc.append(True)

        # Retry with fresh address if no keyword matches
        if (
                not any(loc)
                and 'address' in geocode.__dict__
        ):
            loc = []
            del geocode.address
            _ = geocode.address
            for name in matches.index:
                remote_class = cls.catalog[name]
                keyword = remote_class.prototype.keyword
                dropword = remote_class.prototype.dropword

                # Check dropwords
                if dropword:
                    if isinstance(dropword, str):
                        dropwords = dropword,
                    else:
                        dropwords = dropword

                    if any(
                            word.casefold() in geocode.address.casefold()
                            for word in dropwords
                    ):
                        loc.append(False)
                        continue

                # Check keywords
                if keyword:
                    if isinstance(keyword, str):
                        loc.append(keyword.casefold() in geocode.address.casefold())
                    else:
                        append = any(
                            word.casefold() in geocode.address.casefold()
                            for word in keyword
                        )
                        loc.append(append)
                else:
                    loc.append(True)

        if any(loc):
            matches = matches.loc[loc]
        elif 'address' not in geocode.__dict__:
            raise RemoteNotFound(f"Could not geocode location: {item}")
        else:
            logger.warning(
                f'No keyword matches found for {item=} using '
                f'{geocode.address=}; the result may be inaccurate',
            )

        # If single match, return it
        if len(matches) == 1:
            name = matches.index[0]
            return cls.catalog[name].prototype

        # Multiple matches: choose by intersection-over-union
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bboxs = (
                matches
                .intersection(geocode.polygon)
                .area
                .div(matches.area)
            )

        item_name = bboxs.idxmax()
        if len(bboxs) > 1:
            remote_class = cls.catalog[item_name]
            keyword = remote_class.prototype.keyword
            logger.info(
                f'Found multiple remotes for the location, in descending IOU: '
                f'{bboxs.sort_values(ascending=False).index.tolist()} and '
                f'chose {item_name} ({keyword})'
            )

        if isinstance(item_name, str):
            if item_name not in cls.catalog:
                raise RemoteNotFound(
                    f"Remote '{item_name}' found but not in catalog"
                )
            prototype = cls.catalog[item_name]
        else:
            raise TypeError(f'Invalid type {type(item_name)} for {item_name}')

        out = prototype.__class__()
        return out

    @classmethod
    def from_inferred(cls, item) -> Self:
        """
        Infer the appropriate Remote instance from various input types.

        >>> Remote.from_inferred('https://example.com/{z}/{x}/{y}.png')
        >>> Remote.from_inferred('nyc')
        >>> Remote.from_inferred('New York City')
        >>> Remote.from_inferred([...])  # bbox coordinates
        """
        if isinstance(item, str):
            return cls.from_str(item)

        return cls.from_location(item)

    @classmethod
    def from_str(cls, value: str) -> Self:
        """
        Parse a string into a Remote instance by delegating to specialized methods.
        """
        # Try registered name first
        try:
            return cls.from_name(value)
        except SourceParseError:
            ...
        try:
            return cls.from_url(value)
        except SourceParseError:
            ...

        return cls.from_location(value)

    @classmethod
    def from_url(cls, value: str) -> Self:
        """
        Parse a URL string into a Remote instance.
        Placeholder for future URL parsing implementation.
        """
        raise SourceParseError('URL parsing not implemented yet')

    @classmethod
    def from_name(cls, name: str) -> Self:
        """
        Retrieve a Remote prototype by its registered name.
        """
        if name in cls.catalog:
            remote_class = cls.catalog[name]
            return remote_class.prototype
        else:
            msg = f'No Remote source registered with name: {name!r}'
            raise InvalidRemoteName(msg)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        prototype = cls.prototype
        if (
                ABC not in cls.__bases__
                and prototype.enabled
        ):
            cls.catalog[prototype.name] = prototype


if __name__ == '__main__':
    from tile2net.grid.source.arcgis import *
    from tile2net.grid.source.misc import *

    print("Testing Remote.from_inferred()...")
    print("=" * 80)

    # test by registered name
    print("\n1. Testing by registered name:")
    if NewYorkCity.prototype.enabled:
        assert isinstance(Remote.from_inferred('nyc'), NewYorkCity)
        print("  ✓ 'nyc' -> NewYorkCity")
    if NewYork.prototype.enabled:
        assert isinstance(Remote.from_inferred('ny'), NewYork)
        print("  ✓ 'ny' -> NewYork")
    if Massachusetts.prototype.enabled:
        assert isinstance(Remote.from_inferred('ma'), Massachusetts)
        print("  ✓ 'ma' -> Massachusetts")
    if KingCountyWashington.prototype.enabled:
        assert isinstance(Remote.from_inferred('king'), KingCountyWashington)
        print("  ✓ 'king' -> KingCountyWashington")
    if LosAngeles.prototype.enabled:
        assert isinstance(Remote.from_inferred('la'), LosAngeles)
        print("  ✓ 'la' -> LosAngeles")
    if NewJersey.prototype.enabled:
        assert isinstance(Remote.from_inferred('nj'), NewJersey)
        print("  ✓ 'nj' -> NewJersey")
    if SpringHillTN.prototype.enabled:
        assert isinstance(Remote.from_inferred('sh_tn'), SpringHillTN)
        print("  ✓ 'sh_tn' -> SpringHillTN")
    if Virginia.prototype.enabled:
        assert isinstance(Remote.from_inferred('va'), Virginia)
        print("  ✓ 'va' -> Virginia")
    if AlamedaCounty.prototype.enabled:
        assert isinstance(Remote.from_inferred('al'), AlamedaCounty)
        print("  ✓ 'al' -> AlamedaCounty")
    if SanFrancisco2024.prototype.enabled:
        assert isinstance(Remote.from_inferred('sf2024'), SanFrancisco2024)
        print("  ✓ 'sf2024' -> SanFrancisco2024")
    if SanFrancisco2023.prototype.enabled:
        assert isinstance(Remote.from_inferred('sf2023'), SanFrancisco2023)
        print("  ✓ 'sf2023' -> SanFrancisco2023")
    if Maine.prototype.enabled:
        assert isinstance(Remote.from_inferred('maine'), Maine)
        print("  ✓ 'maine' -> Maine")

    # test by city/state name
    print("\n2. Testing by location (city/state names):")
    if NewYorkCity.prototype.enabled:
        assert isinstance(Remote.from_inferred('New York City'), NewYorkCity)
        print("  ✓ 'New York City' -> NewYorkCity")
        assert isinstance(Remote.from_inferred('City of New York'), NewYorkCity)
        print("  ✓ 'City of New York' -> NewYorkCity")
    if Massachusetts.prototype.enabled:
        assert isinstance(Remote.from_inferred('Massachusetts'), Massachusetts)
        print("  ✓ 'Massachusetts' -> Massachusetts")
    if KingCountyWashington.prototype.enabled:
        assert isinstance(Remote.from_inferred('King County, Washington'), KingCountyWashington)
        print("  ✓ 'King County, Washington' -> KingCountyWashington")
        assert isinstance(Remote.from_inferred('King County, Washington'), KingCountyWashington)
        print("  ✓ 'King County' -> KingCountyWashington")
    if LosAngeles.prototype.enabled:
        assert isinstance(Remote.from_inferred('Los Angeles'), LosAngeles)
        print("  ✓ 'Los Angeles' -> LosAngeles")
    if NewJersey.prototype.enabled:
        assert isinstance(Remote.from_inferred('New Jersey'), NewJersey)
        print("  ✓ 'New Jersey' -> NewJersey")
    if SpringHillTN.prototype.enabled:
        assert isinstance(Remote.from_inferred('Spring Hill, Tennessee'), SpringHillTN)
        print("  ✓ 'Spring Hill, Tennessee' -> SpringHillTN")
    if Virginia.prototype.enabled:
        assert isinstance(Remote.from_inferred('Virginia'), Virginia)
        print("  ✓ 'Virginia' -> Virginia")
    if Maine.prototype.enabled:
        assert isinstance(Remote.from_inferred('Maine'), Maine)
        print("  ✓ 'Maine' -> Maine")

    # test by specific addresses/cities within coverage areas
    print("\n3. Testing by specific addresses within coverage:")
    if Maine.prototype.enabled:
        assert isinstance(Remote.from_inferred('Portland, Maine'), Maine)
        print("  ✓ 'Portland, Maine' -> Maine")
    if NewJersey.prototype.enabled:
        assert isinstance(Remote.from_inferred('New Brunswick, New Jersey'), NewJersey)
        print("  ✓ 'New Brunswick, New Jersey' -> NewJersey")
        assert isinstance(Remote.from_inferred('Jersey City'), NewJersey)
        print("  ✓ 'Jersey City' -> NewJersey")
        assert isinstance(Remote.from_inferred('Hoboken'), NewJersey)
        print("  ✓ 'Hoboken' -> NewJersey")
    if AlamedaCounty.prototype.enabled:
        assert isinstance(Remote.from_inferred('Berkeley, California'), AlamedaCounty)
        print("  ✓ 'Berkeley, California' -> AlamedaCounty")
        assert isinstance(Remote.from_inferred('Fremont, California'), AlamedaCounty)
        print("  ✓ 'Fremont, California' -> AlamedaCounty")
        assert isinstance(Remote.from_inferred('Oakland, California'), AlamedaCounty)
        print("  ✓ 'Oakland, California' -> AlamedaCounty")
    if SanFrancisco2024.prototype.enabled:
        assert isinstance(Remote.from_inferred('San Francisco, California'), SanFrancisco2024)
        print("  ✓ 'San Francisco, California' -> SanFrancisco2024")
    # test by URL (creates new Remote instance, not prototype)
    print("\n4. Testing by URL (generic Remote from URL):")
    url1 = 'https://tiles.example.com/{z}/{x}/{y}.png'
    remote1 = Remote.from_inferred(url1)
    assert isinstance(remote1, Remote)
    assert remote1.format == url1
    assert remote1.scheme == 'https'
    assert remote1.netloc == 'tiles.example.com'
    assert remote1.extension == 'png'
    print(f"  ✓ '{url1}' -> Remote(format={remote1.format})")

    url2 = 'https://example.com/tiles/{z}/{x}/{y}.jpg'
    remote2 = Remote.from_inferred(url2)
    assert remote2.format == url2
    assert remote2.extension == 'jpg'
    print(f"  ✓ '{url2}' -> Remote(extension={remote2.extension})")

    url3 = 'https://server.com/path/{{z}}/{{x}}/{{y}}.png'
    remote3 = Remote.from_inferred(url3)
    assert remote3.format == url3
    print(f"  ✓ URL with {{{{placeholders}}}} -> Remote")
    # test that invalid names/locations raise errors
    print("\n6. Testing error cases:")
    try:
        Remote.from_inferred('nonexistent_name')
        assert False, "Should have raised SourceParseError"
    except SourceParseError:
        print("  ✓ Invalid name raises SourceParseError")

    try:
        Remote.from_inferred('https://example.com/no/placeholders.png')
        assert False, "Should have raised SourceParseError"
    except SourceParseError:
        print("  ✓ URL without placeholders raises SourceParseError")

    try:
        Remote.from_inferred('Antarctica')
        assert False, "Should have raised RemoteNotFound"
    except RemoteNotFound:
        print("  ✓ Location without coverage raises RemoteNotFound")

    print("\n" + "=" * 80)
    print("All Remote.from_inferred() tests passed! ✓")
    print("=" * 80)


