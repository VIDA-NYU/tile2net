from __future__ import annotations

import copy
import os
import os.path
import shutil
import tempfile
import textwrap
import threading
import warnings
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import singledispatchmethod
from pathlib import Path
from typing import *
from typing import Union

import certifi
import imageio.v3 as iio
import pandas as pd
import pyarrow as pa
import shapely.geometry
import yaml
from geopandas import GeoDataFrame, GeoSeries
from tqdm import tqdm

from tile2net.grid.cfg import cfg
from tile2net.grid.geocode import GeoCode
from tile2net.grid.source.name2prototype import Name2Prototype, _Name2Prototype
from tile2net.grid.source.name2base import _Name2Base
from tile2net.grid.source.exceptions import InvalidLocation, InvalidRemoteName, SourceParseError, RemoteNotFound
from tile2net.grid.source.prototype import Prototype
from tile2net.grid.source.remote2coverage import Remote2Coverage
from tile2net.grid.source.source import Source
from tile2net.grid.util import recursion_block
from tile2net.logger import logger

tls = threading.local()

if False:
    # for some reason, my IDE keeps saying SourceParseError is unused, and then clearing it;
    # this is a workaround
    _ = SourceParseError, GeoSeries

@dataclass
class MatchResult:
    """Result of matching a single feature key against criteria."""
    needles: tuple[str, ...]
    haystack: str
    found: bool


class Remote(
    Source,
    ABC
):
    """Base class for remote tile sources"""

    server: str

    def __repr__(self) -> str:
        attributes = []
        try:
            attributes.append(f'name={self.name!r}')
        except AttributeError:
            ...
        try:
            attributes.append(f'url={self.server!r}')
        except AttributeError:
            ...
        return f"{self.__class__.__qualname__}(\n    " + ",\n    ".join(attributes) + "\n)"

    @Name2Prototype
    def name2prototype(self) -> dict[str, Remote]:
        """
        Catalog, mapping the name to the prototype instance, for each enabled Remote subclass.
        The public catalog is assembled using the private catalog and the `servers.yaml`:
        >>> Name2Prototype.__get__
        """

    @Remote2Coverage
    def remote2coverage(self):
        """
        Spatial coverage GeoSeries for all registered remotes.
        Can be called to find the best matching remote for a location.
        >>> Remote2Coverage.__get__
        """

    @Prototype
    def prototype(self):
        """
        Prototype instance for each class.
        >>> Prototype.__get__
        """

    enabled: bool = True
    """
    Whether this remote source is enabled for use.
    Set False to disable it from being constructed.
    """

    name: str = ''
    """Short name of the remote source, e.g. `nyc`."""

    original: str
    """Original URL string as provided, before parsing."""

    template: str
    """URL template with {x}, {y}, {z} placeholders for tile coordinates."""

    scheme: str
    """URL scheme (http or https)."""

    path: str
    """URL path component."""

    extension: str
    """File extension without the leading dot (e.g., 'png', 'jpg')."""

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

    dropword: dict[str, Union[str, tuple[str, ...]]] = None
    """
    Dropword dict is the reverse of keyword. If a geocode's features
    match all dropword criteria, the remote is excluded.

    Uses the same format as keyword.
    """

    year: int
    """Year of the data"""

    dimension: int = 256
    """Default dimension of the remote grid, e.g. 256 pixels."""

    zoom: int = 19
    """Default XYZ zoom level for the remote."""

    coverage: GeoSeries | GeoDataFrame
    """Geographic coverage area of the remote source."""

    @property
    def url(self) -> pd.Series:
        """Generate URLs for all tiles in the attached grid."""
        grid = self.grid
        key = f'{self.__name__}.url'
        if key not in grid.columns:
            template = self.template
            it = zip(grid.xtile.values, grid.ytile.values)
            obj = (
                template.format(z=grid.zoom, y=ytile, x=xtile)
                for xtile, ytile in it
            )
            arr = pa.array(obj, type=pa.string(), size=len(grid))
            mapper = {pa.string(): pd.ArrowDtype(pa.string())}.get
            out = arr.to_pandas(types_mapper=mapper)
            out.index = grid.index
            grid[key] = out

        return grid[key]

    @url.setter
    def url(self, value: pd.Series):
        """Set the URLs for all tiles in the attached grid."""
        grid = self.grid
        key = f'{self.__name__}.url'
        grid[key] = value

    @url.deleter
    def url(self):
        """Delete the URLs for all tiles in the attached grid."""
        grid = self.grid
        key = f'{self.__name__}.url'
        try:
            del grid[key]
        except KeyError:
            ...

    def download_one(self) -> pd.Series:
        """
        Download a single tile for testing purposes.
        Uses context manager to avoid caching the static column.
        """
        grid = self.grid
        with grid.file._static_peek():
            paths = grid.file.static
            urls = self.url

            if paths.empty or urls.empty:
                return paths
            if len(paths) != len(urls):
                raise ValueError("paths and urls must be equal length")

            for p in {Path(p).parent for p in paths}:
                p.mkdir(parents=True, exist_ok=True)

            exists = (
                paths
                .map(os.path.exists)
                .to_numpy(bool)
            )

            if exists.any():
                logger.info("One file already exists, skipping download.")
                return paths

            mapping = dict(zip(paths.array, urls.array))
            if not mapping:
                return paths

            logger.info(
                f'Downloading one {grid.__class__.__qualname__}.{grid.file.static.name} '
                f'from {self.name}'
            )

            not_found: list[Path] = []
            failed: list[Path] = []
            succeeded: list[Path] = []

            def _fetch_write(
                    path: Path,
                    url: str,
            ) -> str:
                """Returns 'success', 'not_found', or 'failed'"""
                try:
                    with tls.session.get(url, stream=True, timeout=(3, 30)) as resp:
                        status = resp.status_code
                        if status in (404, 403):
                            not_found.append(path)
                            return 'not_found'
                        if status != 200:
                            failed.append(path)
                            return 'failed'

                        fd, tmp_path = tempfile.mkstemp(
                            dir=path.parent,
                            suffix=".part",
                        )
                        os.close(fd)
                        tmp = Path(tmp_path)

                        try:
                            with tmp.open("wb") as fh:
                                for chunk in resp.iter_content(1 << 15):
                                    if not chunk:
                                        continue
                                    fh.write(chunk)

                            try:
                                iio.imread(tmp)
                            except ValueError:
                                tmp.unlink(missing_ok=True)
                                failed.append(path)
                                return 'failed'

                            shutil.move(tmp, path)
                            succeeded.append(path)
                            return 'success'

                        except Exception:
                            tmp.unlink(missing_ok=True)
                            failed.append(path)
                            return 'failed'

                except Exception:
                    failed.append(path)
                    return 'failed'

            pool_size = min(4, len(mapping))
            with ThreadPoolExecutor(
                    max_workers=pool_size,
                    initializer=self._init_net_worker,
            ) as pool:

                futures = {
                    pool.submit(_fetch_write, Path(p), u): Path(p)
                    for p, u in mapping.items()
                }

                pbar = tqdm(
                    total=len(futures),
                    desc="Downloading one image for metadata",
                    unit=" attempts",
                    mininterval=1,
                )

                for fut in as_completed(futures):
                    result = fut.result()
                    pbar.set_postfix(
                        success=len(succeeded),
                        not_found=len(not_found),
                        failed=len(failed),
                        refresh=False,
                    )
                    pbar.update(1)

                    if result == 'success':
                        for f in futures:
                            f.cancel()
                        pbar.close()
                        logger.info("Downloaded one file successfully.")
                        return paths

                pbar.close()

            msg = (
                f'Failed to download any file: '
                f'{len(not_found)} not found, {len(failed)} failed'
            )
            raise FileNotFoundError(msg)

    @recursion_block
    def download(
            self,
            retry: bool = True,
            force: bool = False,
            max_workers: int = 64,
    ):
        """
        Download all tiles from this remote source to the input directory.

        Args:
            retry:
                Retry failed downloads once
            force:
                Re-download existing files
            max_workers:
                Maximum concurrent download threads
        """
        grid = self.grid
        paths = grid.file.static
        urls = self.url

        if paths.empty or urls.empty:
            return paths
        if len(paths) != len(urls):
            raise ValueError("paths and urls must be equal length")

        for p in {Path(p).parent for p in paths}:
            p.mkdir(parents=True, exist_ok=True)

        exists = (
            paths
            .map(os.path.exists)
            .to_numpy(bool)
        )

        if exists.all() and not force:
            msg = (
                f'All {len(paths):,} '
                f'{grid.__class__.__qualname__}.{grid.file.static.name} '
                f'on disk at {grid.outdir.source.static.dir}. '
            )
            logger.info(msg)
            return paths

        if not force:
            paths = paths[~exists]
            urls = urls[~exists]

        mapping = dict(zip(paths.array, urls.array))
        if not mapping:
            return paths

        msg = (
            f'Downloading {len(mapping):,} '
            f'{grid.__class__.__qualname__}.{grid.file.static.name} '
            f'from {self.name} to \n\t{grid.outdir.source.static.dir} '
        )
        logger.info(msg)

        pool_size = min(max_workers, len(mapping))
        not_found: list[Path] = []
        failed: list[Path] = []
        succeeded: list[Path] = []

        def _fetch_write(
                path: Path,
                url: str,
        ) -> str:
            """Returns 'success', 'not_found', or 'failed'"""
            try:
                with tls.session.get(url, stream=True, timeout=(3, 30)) as resp:
                    status = resp.status_code
                    if status in (404, 403):
                        not_found.append(path)
                        return 'not_found'
                    if status != 200:
                        failed.append(path)
                        return 'failed'

                    fd, tmp_path = tempfile.mkstemp(
                        dir=path.parent,
                        suffix=".part",
                    )
                    os.close(fd)
                    tmp = Path(tmp_path)

                    try:
                        with tmp.open("wb") as fh:
                            for chunk in resp.iter_content(1 << 15):
                                if not chunk:
                                    continue
                                fh.write(chunk)

                        try:
                            iio.imread(tmp)
                        except ValueError:
                            tmp.unlink(missing_ok=True)
                            failed.append(path)
                            return 'failed'

                        shutil.move(tmp, path)
                        succeeded.append(path)
                        return 'success'

                    except Exception:
                        tmp.unlink(missing_ok=True)
                        failed.append(path)
                        return 'failed'

            except Exception:
                failed.append(path)
                return 'failed'

        with ThreadPoolExecutor(
                max_workers=pool_size,
                initializer=self._init_net_worker,
        ) as pool:

            futures = {
                pool.submit(_fetch_write, Path(p), u): Path(p)
                for p, u in mapping.items()
            }

            pbar = tqdm(
                total=len(futures),
                desc="Downloading",
                unit=" tiles",
                mininterval=10,
            )

            for fut in as_completed(futures):
                result = fut.result()
                pbar.set_postfix(
                    success=len(succeeded),
                    not_found=len(not_found),
                    failed=len(failed),
                    refresh=False,
                )
                pbar.update(1)

            pbar.close()

        total_downloaded = len(succeeded)
        total_failed = len(failed) + len(not_found)

        if total_downloaded == 0:
            msg = (
                f'All {total_failed:,} downloads failed: '
                f'{len(not_found)} not found, {len(failed)} errors'
            )
            raise FileNotFoundError(msg)

        if not_found:
            black = grid.static.black
            logger.warning(
                "%d tile(s) returned 404/403 – linking placeholder.",
                len(not_found)
            )
            for p in not_found:
                try:
                    os.symlink(black, p)
                except (OSError, NotImplementedError):
                    shutil.copy2(black, p)

        if failed:
            if retry:
                logger.error("%d tile(s) failed; retrying once.", len(failed))
                return self.download(
                    retry=False,
                    force=False,
                    max_workers=max_workers,
                )
            raise FileNotFoundError(f"{len(failed):,} files failed.")

        msg = (
            f"Downloaded {total_downloaded:,} tiles "
            f"({len(not_found)} not found, {len(failed)} failed)"
        )
        logger.info(msg)
        return grid.file.static

    def _make_session(
            self,
            *,
            pool: int,
            retry: 'Retry',
    ) -> requests.Session:
        """
        One reusable Session with a bounded socket-pool.
        `pool_block=True` ⇒ if all connections are busy the thread *waits*
        instead of opening a new source port (prevents TIME_WAIT storms).
        """
        s = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool,
            pool_maxsize=pool,
            pool_block=True,
            max_retries=retry,
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "grid"})
        s.verify = certifi.where()
        return s

    def _init_net_worker(self) -> None:
        """
        Run once per thread → attach one Session to thread-local storage.
        Only a GET Session is needed now that the HEAD phase is gone.
        """
        # do not move to module level import
        from urllib3.util import Retry
        retry_get = Retry(
            total=3,
            backoff_factor=0.4,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        tls.session = self._make_session(pool=8, retry=retry_get)

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
            out = prototype.__class__()

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
                raise RemoteNotFound(
                    f"Remote '{item_name}' found but not in name2prototype"
                )
            prototype = cls.name2prototype[item_name]
        else:
            raise TypeError(f'Invalid type {type(item_name)} for {item_name}')

        msg = f'Matched remote "{prototype}" for location "{geocode.passed}"'
        logger.info(msg)
        out = prototype.__class__()
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
            name: str = None
    ) -> Self:
        """
        Parse a URL string into a Remote instance.
        Placeholder for future URL parsing implementation.
        """
        if name:
            try:
                return cls._name2base[name]
            except KeyError as e:
                msg = f'No Remote subclass registered with name: {name!r}'
                raise SourceParseError(msg) from e

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

    @singledispatchmethod
    @classmethod
    def from_yaml(cls, obj: Any) -> Union[Self, dict[str, Self]]:
        """
        If a YAML path/str is passed, returns a dict, mapping names to Remote instances.
        If a dict is passed, returns a singular Remote instance based on its metadata.
        """
        msg = f'YAML input must be a dict, str, or Path, not {type(obj)}'
        raise SourceParseError(msg)

    @from_yaml.register
    def _(cls, obj: str) -> dict[str, Self]:
        # Delegate string handling to the Path handler
        return cls.from_yaml(Path(obj))

    @from_yaml.register
    def _(cls, obj: Path) -> dict[str, Self]:
        # File Loading Logic
        path = obj.expanduser().resolve()

        with path.open('r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            msg = f'YAML content must parse to a dict, got {type(data)}'
            raise SourceParseError(msg)

        out = {
            name: cls.from_yaml(dct | {'name': name})
            for name, dct in data.items()
            if dct.get('enabled', True)
        }
        return out

    @from_yaml.register
    def _(cls, obj: dict) -> Self:
        try:
            server = obj['server']
        except KeyError as e:
            msg = 'YAML must contain a "server" key for Remote sources.'
            raise SourceParseError(msg) from e

        try:
            name = obj['name']
        except KeyError as e:
            msg = 'YAML must contain a "name" key for Remote sources.'
            raise SourceParseError(msg) from e

        if 'base' in obj:
            base = obj['base']
            try:
                cls = cls._name2base[base]
            except KeyError as e:
                msg = f'No Remote subclass registered with name: {base!r}'
                raise SourceParseError(msg) from e

        out = cls.from_server(server, name=name)

        for key, value in obj.items():
            setattr(out, key, value)

        if 'coverage' in obj:
            coverage = obj['coverage']
            try:
                poly = wkt.loads(coverage)
            except Exception as e:
                msg = f'Could not parse coverage WKT: {coverage!r}'
                raise SourceParseError(msg) from e

            crs = obj.get('crs', 'EPSG:4326')
            coverage = GeoSeries([poly], crs=crs).to_crs(4326)
            out.coverage = coverage

        return out

    def __init_subclass__(cls: type[Remote], **kwargs):
        super().__init_subclass__()
        prototype = cls.prototype
        if (
                ABC not in cls.__bases__
                and prototype
        ):
            name = prototype.name or prototype.__class__.__qualname__
            cls._name2prototype[name] = prototype
        if cls.from_server.__name__ in cls.__dict__:
            cls._name2base[cls.__name__] = cls

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
            actual_lower = actual.casefold()

            if isinstance(expected, tuple):
                needles = expected
            else:
                needles = (expected,)

            found = any(
                needle.casefold() in actual_lower
                or actual_lower in needle.casefold()
                for needle in needles
            )

            results[key] = MatchResult(
                needles=needles,
                haystack=actual,
                found=found
            )

        return results

    @_Name2Base
    def _name2base(self) -> dict[str, type[Remote]]:
        """
        Maps the name of a Remote subclass to the subclass itself.
        >>> _Name2Base.data

        Populated as each Remote subclass is defined:
        >>> Remote.__init_subclass__

        Used for instantiating Remote instances based on URLs:
        >>> Remote.from_server
        """

    @_Name2Prototype
    def _name2prototype(self) -> dict[str, Remote]:
        """
        Private version of the name2prototype catalog.

        It's a dict mapping registered names to Remote prototype instances:
        >>> _Name2Prototype.data

        The private catalog is populated as each Remote subclass is defined:
        >>> Remote.__init_subclass__
        """


if __name__ == '__main__':
    from tile2net.grid.source.arcgis import *
    from tile2net.grid.source.misc import *
    from tile2net.grid.source.vexcel import *

    if NewYorkCity.prototype.enabled:
        assert isinstance(
            Remote.from_inferred('Central Park, New York'),
            NewYorkCity
        )
        print('  ✓ Central Park, New York -> NewYorkCity')

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
    # print("\n4. Testing by URL (generic Remote from URL):")
    # url1 = 'https://tiles.example.com/{z}/{x}/{y}.png'
    # remote1 = Remote.from_inferred(url1)
    # assert isinstance(remote1, Remote)
    # assert remote1.template == url1
    # assert remote1.scheme == 'https'
    # assert remote1.extension == 'png'
    # print(f"  ✓ '{url1}' -> Remote(format={remote1.template})")
    #
    # url2 = 'https://example.com/tiles/{z}/{x}/{y}.jpg'
    # remote2 = Remote.from_inferred(url2)
    # assert remote2.template == url2
    # assert remote2.extension == 'jpg'
    # print(f"  ✓ '{url2}' -> Remote(extension={remote2.extension})")
    #
    # url3 = 'https://server.com/path/{{z}}/{{x}}/{{y}}.png'
    # remote3 = Remote.from_inferred(url3)
    # assert remote3.template == url3
    # print(f"  ✓ URL with {{{{placeholders}}}} -> Remote")
    # # test that invalid names/locations raise errors
    # print("\n6. Testing error cases:")
    # try:
    #     Remote.from_inferred('nonexistent_name')
    #     assert False, "Should have raised SourceParseError"
    # except SourceParseError:
    #     print("  ✓ Invalid name raises SourceParseError")
    #
    # try:
    #     Remote.from_inferred('https://example.com/no/placeholders.png')
    #     assert False, "Should have raised SourceParseError"
    # except SourceParseError:
    #     print("  ✓ URL without placeholders raises SourceParseError")
    #
    # try:
    #     Remote.from_inferred('Antarctica')
    #     assert False, "Should have raised RemoteNotFound"
    # except RemoteNotFound:
    #     print("  ✓ Location without coverage raises RemoteNotFound")
    #
    # print("\n" + "=" * 80)
    # print("All Remote.from_inferred() tests passed! ✓")
    # print("=" * 80)
