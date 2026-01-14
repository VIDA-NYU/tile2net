from __future__ import annotations

import os
import os.path
import shutil
import tempfile
import threading
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from pathlib import Path
from typing import Union

import certifi
import imageio.v3 as iio
import pandas as pd
import pyarrow as pa
import requests
import shapely.geometry
from geopandas import GeoDataFrame
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
import geopandas as gpd


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

tls = threading.local()


class Remote(
    Source,
    ABC
):
    """Base class for remote tile sources"""

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
        >>> Coverage._get
        """

    @Prototype
    def prototype(self):
        """
        Prototype instance for each class.
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
    def template(self) -> str:
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
    def url(self) -> pd.Series:
        """Generate URLs for all tiles in the attached grid."""
        grid = self.ingrid
        key = f'{self.__name__}.url'
        if key not in grid:
            template = self.template
            it = zip(grid.xtile.values, grid.ytile.values)
            obj = (
                template.format(z=grid.zoom, y=ytile, x=xtile)
                for xtile, ytile in it
            )
            arr = pa.array(obj, type=pa.string(), size=len(grid))
            grid[key] = pd.Series(arr.to_pandas(), index=grid.index, dtype='string')

        return grid[key]

    @url.setter
    def url(self, value: pd.Series):
        """Set the URLs for all tiles in the attached grid."""
        grid = self.ingrid
        key = f'{self.__name__}.url'
        grid[key] = value

    @url.deleter
    def url(self):
        """Delete the URLs for all tiles in the attached grid."""
        grid = self.ingrid
        key = f'{self.__name__}.url'
        try:
            del grid[key]
        except KeyError:
            ...

    def download(
            self,
            retry: bool = True,
            force: bool = False,
            max_workers: int = 64,
            one: bool = False
    ):
        """
        Download tiles from this remote source to the input directory.

        Args:
            retry:
                Retry failed downloads once
            force:
                Re-download existing files
            max_workers:
                Maximum concurrent download threads
            one:
                Download only one file for testing
        """
        ingrid = self.ingrid
        paths = ingrid.file.static
        urls = self.url

        if paths.empty or urls.empty:
            return ingrid
        if len(paths) != len(urls):
            raise ValueError("paths and urls must be equal length")

        for p in {Path(p).parent for p in paths}:
            p.mkdir(parents=True, exist_ok=True)

        exists = (
            paths
            .map(os.path.exists)
            .to_numpy(bool)
        )

        if (
                exists.all()
                and not force
        ):
            msg = (
                f'All {len(paths):,} '
                f'{ingrid.__class__.__qualname__}.{ingrid.file.static.name} '
                f'on disk at {ingrid.indir.dir}. '
            )
            logger.info(msg)
            return ingrid

        if not force:
            paths = paths[~exists]
            urls = urls[~exists]

        mapping = dict(zip(paths.array, urls.array))
        if not mapping:
            return ingrid

        msg = (
            f'Downloading {len(mapping):,} '
            f'{ingrid.__class__.__qualname__}.{ingrid.file.static.name} '
            f'from {self.name} to \\n\\t{ingrid.indir.dir} '
        )
        logger.info(msg)

        pool_size = min(max_workers, len(mapping))
        not_found: list[Path] = []
        failed: list[Path] = []
        success = False

        def _fetch_write(
                path: Path,
                url: str,
        ) -> None:
            try:
                with tls.session.get(url, stream=True, timeout=(3, 30)) as resp:
                    status = resp.status_code
                    if status == 404:
                        not_found.append(path)
                        return
                    if status == 403:
                        not_found.append(path)
                        return
                    if status != 200:
                        failed.append(path)
                        return

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
                            return

                        shutil.move(tmp, path)

                    except Exception:
                        tmp.unlink(missing_ok=True)
                        failed.append(path)
                        return

            except Exception:
                failed.append(path)
                return

        with ThreadPoolExecutor(
                max_workers=pool_size,
                initializer=self._init_net_worker,
        ) as pool:

            futures = {
                pool.submit(_fetch_write, Path(p), u)
                for p, u in mapping.items()
            }

            for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Downloading",
                    unit=f" {ingrid.__class__.__name__}.{paths.name}",
                    mininterval=10,
            ):
                fut.result()

                if one:
                    success = True
                    for f in futures:
                        f.cancel()
                    break

        if one and success:
            logger.info("Downloaded one file successfully (one=True).")
            return ingrid

        if len(not_found) + len(failed) == len(ingrid.file.static):
            msg = f'All {len(not_found) + len(failed):,} grid failed to download.'
            raise FileNotFoundError(msg)

        if not_found:
            black = ingrid.static.black
            logger.warning("%d tile(s) returned 404/403 – linking placeholder.", len(not_found))
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
            f"All requested {ingrid.__class__.__name__}.{paths.name} "
            f"on disk at \\n\\t{ingrid.indir.dir} "
        )
        logger.info(msg)

    def _make_session(
            self,
            *,
            pool: int,
            retry: Retry,
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
        # Get all coverage geometries
        matches: GeoSeries = cls.coverage

        if isinstance(item, (gpd.GeoDataFrame, gpd.GeoSeries)):
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
    assert remote1.template == url1
    assert remote1.scheme == 'https'
    assert remote1.netloc == 'tiles.example.com'
    assert remote1.extension == 'png'
    print(f"  ✓ '{url1}' -> Remote(format={remote1.template})")

    url2 = 'https://example.com/tiles/{z}/{x}/{y}.jpg'
    remote2 = Remote.from_inferred(url2)
    assert remote2.template == url2
    assert remote2.extension == 'jpg'
    print(f"  ✓ '{url2}' -> Remote(extension={remote2.extension})")

    url3 = 'https://server.com/path/{{z}}/{{x}}/{{y}}.png'
    remote3 = Remote.from_inferred(url3)
    assert remote3.template == url3
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
