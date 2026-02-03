from __future__ import annotations

import copy
import textwrap
import threading
import warnings
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import *
from typing import Union

import shapely.geometry
import yaml
from geopandas import GeoDataFrame, GeoSeries
from shapely import wkt

from tile2net.geo.geocode import GeoCode
from tile2net.geo.source.source import Source
from tile2net.grid.cfg import cfg
from tile2net.grid.cfg.logger import logger
from tile2net.grid.source import remote
from tile2net.grid.source.exceptions import InvalidLocation, InvalidRemoteName, SourceParseError, RemoteNotFound

tls = threading.local()


@dataclass
class MatchResult:
    """Result of matching a single feature key against criteria."""
    needles: tuple[str, ...]
    haystack: str
    found: bool


class Remote(
    remote.Remote,
    Source,
    ABC,
    base=True,
):
    desc: str = ''
    """Brief description of the remote source."""

    """Base class for remote tile sources"""

    server: str

    enabled: bool = True
    """
    Whether this remote source is enabled for use.
    Set False to disable it from being constructed.
    """

    name: str = ''
    """Short name of the remote source, e.g. `nyc`."""

    keyword: dict[str, Union[str, tuple[str, ...]]] = None
    """
    Keyword dict for matching geocode features to resolve discrepancies.
    Keys are feature names (e.g., 'state', 'city'), values are either:
      - A single string to match
      - A tuple of acceptable strings (any one must match)

    Example:
        keyword = dict(
            state=('New York', 'NY'),
            city='New York',
        )

    All key-value pairs must match for the remote to be selected.
    """

    zoom: int = 19
    """Default XYZ zoom level for the remote."""

    coverage: GeoSeries | GeoDataFrame
    """Geographic coverage area of the remote source."""

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

        if isinstance(item, (GeoDataFrame, GeoSeries)):
            infer = (
                item.geometry
                .iat[0]
                .centroid
            )
        else:
            infer = item

        try:
            geocode = GeoCode.from_inferred(infer)
            _ = geocode.polygon
        except Exception as exc:
            msg = f"Could not geocode location: {item} \n\t{exc}"
            raise InvalidLocation(msg) from exc

        return cls.from_geocode(geocode)

    @classmethod
    def from_inferred(cls, item) -> Self:
        """
        Infer the appropriate Remote instance from various input types.

        >>> Remote.from_inferred('nyc')
        >>> Remote.from_inferred('New York City')
        >>> Remote.from_inferred((308800, 393888, 308896, 394048), zoom=20)
        >>> Remote.from_inferred((-73.9730, 40.7648, -73.9712, 40.7665))
        """
        if isinstance(item, GeoCode):
            return cls.from_geocode(item)

        if isinstance(item, str):
            return cls.from_str(item)

        return cls.from_location(item)

    @classmethod
    def from_geocode(cls, geocode: GeoCode) -> Self:
        """
        Creates a Remote instance by finding the best matching coverage for a given geocoded location.

        This method searches through available remote coverages to find one that intersects with
        the provided geocode's polygon. When multiple matches are found, it uses keyword matching
        and intersection-over-union (IOU) calculations to determine the most appropriate remote.
        If the initial geocode doesn't match any coverage, the method attempts to recompute the
        address and retry the search.

        Parameters
        ----------
        geocode : GeoCode
            A geocoded location object containing polygon geometry, address information, and
            geographic features used for matching against"""

        matches: GeoSeries = cls.remote2coverage
        # Find intersecting coverages
        loc = matches.intersects(geocode.polygon)
        if (
                not loc.any()
                and 'address' in geocode.__dict__
        ):
            # User may have been lazy; recompute polygon
            # e.g. user just passed "Springfield" instead of "Springfield, Illinois"
            del geocode.address
            _ = geocode.address
            del geocode.nwse
            del geocode.wsen
            del geocode.polygon
            loc = matches.intersects(geocode.polygon)
            if not loc.any():
                msg = (
                    f'No remote coverage found for location: "{geocode.passed}" \n\t'
                    f'Geocoded as: "{geocode.address}" \n\t'
                    f'Please be more specific.'
                )
                raise RemoteNotFound(msg)
        matches = matches.loc[loc]

        if cfg.download.use_tags:

            # resolve discrepancies using keywords
            tags = geocode.tags
            loc = []
            for name in matches.index:
                prototype = cls.name2prototype[name]
                keyword = prototype.prototype.keyword
                dropword = prototype.prototype.dropword

                excluded = False

                if dropword:
                    results = cls._match_tags(dropword, tags)
                    if any(result.found for result in results.values()):
                        mismatches = [
                            (
                                f"{key}={result.haystack!r} matches dropword "
                                f"{result.needles!r}; remote excluded."
                            )
                            for key, result in results.items()
                            if result.found
                        ]
                        logger.info(
                            f"Excluding remote '{name}' ({prototype.__class__.__name__}): "
                            f"dropword matched on {', '.join(mismatches)}.\n\t"
                            f"To force this Source, pass `source={prototype.name}`"
                        )
                        excluded = True

                if keyword:
                    results = cls._match_tags(keyword, tags)
                    if any(not result.found for result in results.values()):
                        mismatches = [
                            (
                                f"{key}: expected at least one of "
                                f"{result.needles!r}, in {result.haystack!r}; "
                                f"not found"
                            )
                            for key, result in results.items()
                            if not result.found
                        ]
                        logger.info(
                            f"Excluding remote '{name}' ({prototype.__class__.__name__}): "
                            f"keyword mismatch on {', '.join(mismatches)}.\n\t"
                            f"To force this Source, pass `source=\"{prototype.name}\"`"
                        )
                        excluded = True

                loc.append(not excluded)

            # Retry with fresh address if no keyword matches
            if (
                    not any(loc)
                    and 'address' in geocode.__dict__
            ):
                loc = []
                del geocode.address
                _ = geocode.address
                tags = geocode.tags
                for name in matches.index:
                    prototype = cls.name2prototype[name]
                    keyword = prototype.keyword
                    dropword = prototype.dropword

                    excluded = False

                    if dropword:
                        results = cls._match_tags(dropword, tags)
                        if any(
                                result.found
                                for result in results.values()
                        ):
                            excluded = True

                    if keyword:
                        results = cls._match_tags(keyword, tags)
                        if any(
                                not result.found
                                for result in results.values()
                        ):
                            excluded = True

                    loc.append(not excluded)

            matches = matches.loc[loc]
        if matches.empty:
            raise RemoteNotFound(
                f'No remote coverage found for location: "{geocode.passed}" \n\t'
                f'Geocoded as: "{geocode.address}" \n\t',
            )
        elif 'address' not in geocode.__dict__:
            raise RemoteNotFound(f"Could not geocode location: {geocode.passed}")
        # else:
        # logger.warning(
        #     f'No keyword matches found for {geocode.passed} using '
        #     f'the geocoded address "{geocode.address}"',
        # )

        # If single match, return it
        if len(matches) == 1:
            name = matches.index[0]
            prototype = cls.name2prototype[name].prototype
            out = copy.copy(prototype)

            proto_log = textwrap.indent(str(prototype), prefix='\t')
            # Move the object to the end and separate with a newline
            msg = (
                f'Matched remote for location "{geocode.passed}":\n'
                f'{proto_log}'
            )
            logger.info(msg)
            return out

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
            prototype = cls.name2prototype[item_name]
            keyword = prototype.prototype.keyword
            logger.info(
                f'Found multiple remotes for the location, in descending IOU: '
                f'{bboxs.sort_values(ascending=False).index.tolist()} and '
                f'chose {item_name} ({keyword})'
            )

        if isinstance(item_name, str):
            if item_name not in cls.name2prototype:
                raise RemoteNotFound(f"Remote '{item_name}' found but not in name2prototype")
            prototype = cls.name2prototype[item_name]
        else:
            raise TypeError(f'Invalid type {type(item_name)} for {item_name}')

        msg = f'Matched remote "{prototype}" for location "{geocode.passed}"'
        logger.info(msg)
        out = copy.copy(prototype)
        return out

    @classmethod
    def from_str(cls, value: str) -> Self:
        """
        Parse a string into a Remote instance by delegating to specialized methods.
        """
        try:
            return cls.from_name(value)
        except SourceParseError:
            ...
        try:
            return cls.from_server(value)
        except SourceParseError:
            ...

        out = cls.from_location(value)
        return out

    @classmethod
    def from_server(
            cls,
            value: str,
            base: str = None
    ) -> Self:
        """
        Parse a URL string into a Remote instance.
        Placeholder for future URL parsing implementation.
        """
        if base:
            try:
                cls = cls._name2base[base]
            except KeyError as e:
                msg = f'No Remote subclass registered with name: {base!r}'
                raise ValueError(msg) from e
            return cls.from_server(value)

        for subclass in cls._name2base.values():
            try:
                return subclass.from_server(value)
            except SourceParseError:
                ...
        msg = f'No Remote subclass could parse URL: {value!r}'
        raise SourceParseError(msg)

    @classmethod
    def from_name(cls, name: str) -> Self:
        """
        Retrieve a Remote prototype by its registered name.
        """
        if name in cls.name2prototype:
            remote_class = cls.name2prototype[name]
            out = copy.copy(remote_class.prototype)
        else:
            msg = f'No Remote source registered with name: {name!r}'
            raise InvalidRemoteName(msg)
        return out

    @classmethod
    def from_yaml(cls, obj: Any) -> Union[Self, dict[str, Self]]:
        """
        If a YAML path/str is passed, returns a dict, mapping names to Remote instances.
        If a dict is passed, returns a singular Remote instance based on its metadata.
        """
        match obj:

            case Path() | str():
                path = (
                    Path(obj)
                    .expanduser()
                    .resolve()
                )
                with path.open('r') as f:
                    data: dict[str, dict] = yaml.safe_load(f)
                if not isinstance(data, dict):
                    msg = f'YAML content must parse to a dict, got {type(data)}'
                    raise SourceParseError(msg)
                out = {
                    name: cls.from_yaml(dct | {'name': name})
                    for name, dct in data.items()
                    if dct.setdefault('enabled', True)
                }
                return out

            case dict():
                try:
                    server = obj['server']
                except KeyError as e:
                    msg = 'YAML must contain a "server" key for Remote sources.'
                    raise SourceParseError(msg) from e

                base = obj.get('base', None)
                if base:
                    # bypass from_server to avoid potential network calls
                    # during simple configuration loading.
                    try:
                        subclass = cls._name2base[base]
                    except KeyError as e:
                        msg = f'No Remote subclass registered with name: {base!r}'
                        raise ValueError(msg) from e

                    out = subclass.__new__(subclass)
                    out.server = server
                else:
                    out = cls.from_server(server, base=base)

                obj.setdefault('enabled', True)

                for key, value in obj.items():
                    setattr(out, key, value)

                if 'coverage' in obj:
                    coverage = obj['coverage']
                    try:
                        poly = wkt.loads(coverage)
                    except shapely.errors.GEOSException as e:
                        msg = f'Could not parse coverage WKT: {coverage!r}'
                        raise SourceParseError(msg) from e

                    crs = obj.get('crs', 'EPSG:4326')
                    # Optimization: Avoid overhead of to_crs if already in target CRS
                    val = GeoSeries([poly], crs=crs)
                    if crs != 'EPSG:4326':
                        val = val.to_crs(4326)
                    out.coverage = val

                if 'enabled' not in obj:
                    out.enabled = True

                out.prototype = out

                return out

            case _:
                msg = f'YAML input must be a dict, str, or Path, not {type(obj)}'
                raise SourceParseError(msg)

    @classmethod
    def _match_tags(
            cls,
            criteria: dict[str, Union[str, tuple[str, ...]]],
            tags: dict[str, str]
    ) -> dict[str, MatchResult]:
        """
        Check if all key-value pairs in criteria match corresponding values in features.

        criteria: dict where values can be either:
            - A single string to match
            - A tuple of strings where any one must match
        features: dict of feature values from geocoding

        Matching is case-insensitive for string values.

        Returns:
            dict mapping feature keys to MatchResult instances, containing only
            keys where both criteria and features exist
        """
        results = {}

        for key, expected in criteria.items():
            if key not in tags:
                continue

            actual = tags[key]
            haystack = actual.casefold()

            if isinstance(expected, str):
                needles = expected,
            elif isinstance(expected, Iterable):
                needles = expected
            else:
                msg = f'Invalid type {type(expected)} for criteria value: {expected!r}'
                raise TypeError(msg)

            found = any(
                needle.casefold() in haystack
                or haystack in needle.casefold()
                for needle in needles
            )

            results[key] = MatchResult(
                needles=needles,
                haystack=actual,
                found=found
            )

        return results

# todo: The import * from this code block was causing my IDE to clean up important imports from
#   the top. Make sure this code block's functionality is reflected in the tests.

# if __name__ == '__main__':
#     from tile2net.geo.source.arcgis import *
#     from tile2net.geo.source.misc import *
#     from tile2net.geo.source.vexcel import *
#
#     if NewYorkCity.prototype.enabled:
#         assert isinstance(
#             Remote.from_inferred('Central Park, New York'),
#             NewYorkCity
#         )
#         print('  ✓ Central Park, New York -> NewYorkCity')
#
#     print("Testing Remote.from_inferred()...")
#     print("=" * 80)
#
#     # test by registered name
#     print("\n1. Testing by registered name:")
#     if NewYorkCity.prototype.enabled:
#         assert isinstance(Remote.from_inferred('nyc'), NewYorkCity)
#         print("  ✓ 'nyc' -> NewYorkCity")
#     if NewYork.prototype.enabled:
#         assert isinstance(Remote.from_inferred('ny'), NewYork)
#         print("  ✓ 'ny' -> NewYork")
#     if Massachusetts.prototype.enabled:
#         assert isinstance(Remote.from_inferred('ma'), Massachusetts)
#         print("  ✓ 'ma' -> Massachusetts")
#     if KingCountyWashington.prototype.enabled:
#         assert isinstance(Remote.from_inferred('king'), KingCountyWashington)
#         print("  ✓ 'king' -> KingCountyWashington")
#     if LosAngeles.prototype.enabled:
#         assert isinstance(Remote.from_inferred('la'), LosAngeles)
#         print("  ✓ 'la' -> LosAngeles")
#     if NewJersey.prototype.enabled:
#         assert isinstance(Remote.from_inferred('nj'), NewJersey)
#         print("  ✓ 'nj' -> NewJersey")
#     if SpringHillTN.prototype.enabled:
#         assert isinstance(Remote.from_inferred('sh_tn'), SpringHillTN)
#         print("  ✓ 'sh_tn' -> SpringHillTN")
#     if Virginia.prototype.enabled:
#         assert isinstance(Remote.from_inferred('va'), Virginia)
#         print("  ✓ 'va' -> Virginia")
#     if AlamedaCounty.prototype.enabled:
#         assert isinstance(Remote.from_inferred('al'), AlamedaCounty)
#         print("  ✓ 'al' -> AlamedaCounty")
#     if SanFrancisco2024.prototype.enabled:
#         assert isinstance(Remote.from_inferred('sf2024'), SanFrancisco2024)
#         print("  ✓ 'sf2024' -> SanFrancisco2024")
#     if SanFrancisco2023.prototype.enabled:
#         assert isinstance(Remote.from_inferred('sf2023'), SanFrancisco2023)
#         print("  ✓ 'sf2023' -> SanFrancisco2023")
#     if Maine.prototype.enabled:
#         assert isinstance(Remote.from_inferred('maine'), Maine)
#         print("  ✓ 'maine' -> Maine")
#
#     # test by city/state name
#     print("\n2. Testing by location (city/state names):")
#     if NewYorkCity.prototype.enabled:
#         assert isinstance(Remote.from_inferred('New York City'), NewYorkCity)
#         print("  ✓ 'New York City' -> NewYorkCity")
#         assert isinstance(Remote.from_inferred('City of New York'), NewYorkCity)
#         print("  ✓ 'City of New York' -> NewYorkCity")
#     if Massachusetts.prototype.enabled:
#         assert isinstance(Remote.from_inferred('Massachusetts'), Massachusetts)
#         print("  ✓ 'Massachusetts' -> Massachusetts")
#     if KingCountyWashington.prototype.enabled:
#         assert isinstance(Remote.from_inferred('King County, Washington'), KingCountyWashington)
#         print("  ✓ 'King County, Washington' -> KingCountyWashington")
#         assert isinstance(Remote.from_inferred('King County, Washington'), KingCountyWashington)
#         print("  ✓ 'King County' -> KingCountyWashington")
#     if LosAngeles.prototype.enabled:
#         assert isinstance(Remote.from_inferred('Los Angeles'), LosAngeles)
#         print("  ✓ 'Los Angeles' -> LosAngeles")
#     if NewJersey.prototype.enabled:
#         assert isinstance(Remote.from_inferred('New Jersey'), NewJersey)
#         print("  ✓ 'New Jersey' -> NewJersey")
#     if SpringHillTN.prototype.enabled:
#         assert isinstance(Remote.from_inferred('Spring Hill, Tennessee'), SpringHillTN)
#         print("  ✓ 'Spring Hill, Tennessee' -> SpringHillTN")
#     if Virginia.prototype.enabled:
#         assert isinstance(Remote.from_inferred('Virginia'), Virginia)
#         print("  ✓ 'Virginia' -> Virginia")
#     if Maine.prototype.enabled:
#         assert isinstance(Remote.from_inferred('Maine'), Maine)
#         print("  ✓ 'Maine' -> Maine")
#
#     # test by specific addresses/cities within coverage areas
#     print("\n3. Testing by specific addresses within coverage:")
#     if Maine.prototype.enabled:
#         assert isinstance(Remote.from_inferred('Portland, Maine'), Maine)
#         print("  ✓ 'Portland, Maine' -> Maine")
#     if NewJersey.prototype.enabled:
#         assert isinstance(Remote.from_inferred('New Brunswick, New Jersey'), NewJersey)
#         print("  ✓ 'New Brunswick, New Jersey' -> NewJersey")
#         assert isinstance(Remote.from_inferred('Jersey City'), NewJersey)
#         print("  ✓ 'Jersey City' -> NewJersey")
#         assert isinstance(Remote.from_inferred('Hoboken'), NewJersey)
#         print("  ✓ 'Hoboken' -> NewJersey")
#     if AlamedaCounty.prototype.enabled:
#         assert isinstance(Remote.from_inferred('Berkeley, California'), AlamedaCounty)
#         print("  ✓ 'Berkeley, California' -> AlamedaCounty")
#         assert isinstance(Remote.from_inferred('Fremont, California'), AlamedaCounty)
#         print("  ✓ 'Fremont, California' -> AlamedaCounty")
#         assert isinstance(Remote.from_inferred('Oakland, California'), AlamedaCounty)
#         print("  ✓ 'Oakland, California' -> AlamedaCounty")
#     if SanFrancisco2024.prototype.enabled:
#         assert isinstance(Remote.from_inferred('San Francisco, California'), SanFrancisco2024)
#         print("  ✓ 'San Francisco, California' -> SanFrancisco2024")
#     # test by URL (creates new Remote instance, not prototype)
#     # print("\n4. Testing by URL (generic Remote from URL):")
#     # url1 = 'https://tiles.example.com/{z}/{x}/{y}.png'
#     # remote1 = Remote.from_inferred(url1)
#     # assert isinstance(remote1, Remote)
#     # assert remote1.template == url1
#     # assert remote1.scheme == 'https'
#     # assert remote1.extension == 'png'
#     # print(f"  ✓ '{url1}' -> Remote(format={remote1.template})")
#     #
#     # url2 = 'https://example.com/tiles/{z}/{x}/{y}.jpg'
#     # remote2 = Remote.from_inferred(url2)
#     # assert remote2.template == url2
#     # assert remote2.extension == 'jpg'
#     # print(f"  ✓ '{url2}' -> Remote(extension={remote2.extension})")
#     #
#     # url3 = 'https://server.com/path/{{z}}/{{x}}/{{y}}.png'
#     # remote3 = Remote.from_inferred(url3)
#     # assert remote3.template == url3
#     # print(f"  ✓ URL with {{{{placeholders}}}} -> Remote")
#     # # test that invalid names/locations raise errors
#     # print("\n6. Testing error cases:")
#     # try:
#     #     Remote.from_inferred('nonexistent_name')
#     #     assert False, "Should have raised SourceParseError"
#     # except SourceParseError:
#     #     print("  ✓ Invalid name raises SourceParseError")
#     #
#     # try:
#     #     Remote.from_inferred('https://example.com/no/placeholders.png')
#     #     assert False, "Should have raised SourceParseError"
#     # except SourceParseError:
#     #     print("  ✓ URL without placeholders raises SourceParseError")
#     #
#     # try:
#     #     Remote.from_inferred('Antarctica')
#     #     assert False, "Should have raised RemoteNotFound"
#     # except RemoteNotFound:
#     #     print("  ✓ Location without coverage raises RemoteNotFound")
#     #
#     # print("\n" + "=" * 80)
#     # print("All Remote.from_inferred() tests passed! ✓")
#     # print("=" * 80)
