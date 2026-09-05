from __future__ import annotations

from pathlib import Path

import copy
import geopy.geocoders
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from contextlib import closing
from geopy.geocoders.base import DEFAULT_SENTINEL
from geopy.location import Location
from typing import Any, Optional, Self

CACHE_DIRECTORY_ENV = 'TILE2NET_CACHE_DIR'
SQLITE_PATH_ENV = 'TILE2NET_NOMINATIM_CACHE'


def _normalize(query: Any) -> str:
    """Stringify a geocode/reverse query into a stable, human-readable form."""
    if hasattr(query, 'latitude') and hasattr(query, 'longitude'):
        return f'{query.latitude},{query.longitude}'
    if isinstance(query, (tuple, list)):
        return ','.join(str(value) for value in query)
    return str(query)


def _to_location(raw: dict) -> Location:
    """Reconstruct a geopy Location from a cached raw response."""
    address = raw.get('display_name', '')
    lat = raw.get('lat')
    lon = raw.get('lon')
    if lat is None or lon is None:
        # the json cache only tracks display_name/boundingbox, so
        # derive the point from the box's center instead of (0, 0)
        south, north, west, east = map(float, raw['boundingbox'])
        lat = (south + north) / 2
        lon = (west + east) / 2
    return Location(address, (float(lat), float(lon)), raw)


class Cache:
    """Descriptor base for a Nominatim result cache layer."""

    parent: Optional[Nominatim]

    def __set_name__(self, owner, name):
        """Record the attribute name and mark self as the class-level singleton."""
        self.__name__ = name
        self._descriptor: Self = self

    def _get_cache(
            self,
            instance: Optional[Nominatim],
            owner,
    ) -> Self:
        """Return the singleton for class-level access, else a copy bound to instance."""
        if instance is None:
            # class-level access (e.g. Nominatim.json.write = True) must
            # mutate the singleton itself, not a throwaway copy
            return self
        out = copy.copy(self)
        out.parent = instance
        return out

    locals().update(__get__=_get_cache)

    def __init__(
            self,
            func=None,
            read=True,
            write=False,
    ):
        """
        Store the read/write toggles for this cache layer.

        read:
            if True, check this cache layer for a cached response before fetching from the network.
        write:
            if True, write fetched responses to this cache layer for future use.
        """
        self.read = read
        self.write = write
        if func is not None:
            self.__name__ = func.__name__
            self.__doc__ = func.__doc__

    def _key(
            self,
            action: str,
            query: Any,
            params: dict,
    ) -> str:
        """Readable canonical key: operation, query, and response-affecting params."""
        parts = [action, _normalize(query)]
        extra = ', '.join(
            f'{key}={value!r}'
            for key, value in sorted(params.items())
            if value not in (None, False)
        )
        if extra:
            parts.append(extra)
        return '|'.join(parts)

    def _write(
            self,
            key: str,
            raw: dict,
    ) -> None:
        """Persist raw under key. Implemented per cache backend."""
        raise NotImplementedError

    def _read(self, key: str) -> Optional[dict]:
        """Fetch a cached raw response by key, or None if absent. Implemented per backend."""
        raise NotImplementedError

    def geocode(
            self,
            query,
            params: Optional[dict] = None,
            exactly_one: bool = True,
            fetch=None,
    ):
        """
        Check this cache layer for a forward-geocode, else fetch and write through.

        query:
            The address, query, or a structured query you wish to geocode.
        params:
            The response-affecting Nominatim.geocode kwargs used to build the
            cache key, e.g. {'language': 'en'}. Defaults to none passed.
        exactly_one:
            If True, return a single Location object. If False, return a list of Location objects.
        fetch:
            A callable that performs the live lookup on a cache miss. Defaults
            to calling Nominatim.geocode on self.parent directly.
        """
        if params is None:
            params = {}
        if fetch is None:
            def fetch():
                return super(Nominatim, self.parent).geocode(
                    query, exactly_one=exactly_one, **params
                )
        return self._through('geocode', query, params, exactly_one, fetch)

    def reverse(
            self,
            query,
            params: Optional[dict] = None,
            exactly_one: bool = True,
            fetch=None,
    ):
        """
        Check this cache layer for a reverse-geocode, else fetch and write through.

        query:
            The coordinates for which you wish to obtain the closest address.
        params:
            The response-affecting Nominatim.reverse kwargs used to build the
            cache key, e.g. {'zoom': 10}. Defaults to none passed.
        exactly_one:
            If True, return a single Location object. If False, return a list of Location objects.
        fetch:
            A callable that performs the live lookup on a cache miss. Defaults
            to calling Nominatim.reverse on self.parent directly.
        """
        if params is None:
            params = {}
        if fetch is None:
            def fetch():
                return super(Nominatim, self.parent).reverse(
                    query, exactly_one=exactly_one, **params
                )
        return self._through('reverse', query, params, exactly_one, fetch)

    def _through(
            self,
            action,
            query,
            params,
            exactly_one,
            fetch,
    ):
        """Shared read-cache-else-fetch-then-write-through logic for geocode/reverse."""
        if not exactly_one:
            # list results aren't cached; this codebase only ever
            # requests exactly_one geocodes/reverses
            return fetch()
        key = self._key(action, query, params)
        if self.read:
            raw = self._read(key)
            if raw is not None:
                return _to_location(raw)
        result = fetch()
        if result is not None and self.write:
            self._write(key, result.raw)
        return result


class JSON(Cache):
    """Repo-tracked cache of known test locations. Not size-bounded."""

    _descriptor: Self
    _fields = ('display_name', 'boundingbox')
    _cache: Optional[dict[str, dict]] = None

    @property
    def path(self) -> Path:
        """Path to the repo-tracked nominatim.json file."""
        if self._descriptor is not self:
            return self._descriptor.path
        path = Path(__file__, '../', 'nominatim.json').resolve()
        self._path = path
        return path

    def __init__(
            self,
            func=None,
            read=True,
            write=False,
    ):
        """Store the read/write toggles for the json cache layer."""
        super().__init__(func, read, write)

    @property
    def cache(self) -> dict[str, dict]:
        """The json cache dict, lazily loaded from disk and shared by all copies."""
        if self._descriptor is not self:
            return self._descriptor.cache
        if self._cache is None:
            if self.path.exists():
                with open(self.path) as f:
                    self._cache = json.load(f)
            else:
                self._cache = {}
        return self._cache

    def _read(self, key: str) -> Optional[dict]:
        """Fetch a cached raw response by key, or None if absent."""
        return self.cache.get(key)

    def _write(
            self,
            key: str,
            raw: dict,
    ) -> None:
        """Trim raw to the tracked fields, store it, and persist the cache to disk."""
        value = {
            field: raw[field]
            for field in self._fields
            if field in raw
        }
        self.cache[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.cache, f, indent=2, sort_keys=True)
            f.write('\n')


class SQLite(Cache):
    """Per-user cache bounded to the `max_entries` most recently used rows."""

    _descriptor: Self

    # Maximum number of rows to keep in the sqlite cache. Older rows are evicted on write.
    max_entries = 100
    # Lock to ensure thread-safe access to the sqlite cache.
    _lock = threading.Lock()

    @classmethod
    def cache_directory(cls) -> Path:
        """User-writable cache directory, mirroring raster.weights.weights_directory."""
        configured = os.environ.get(CACHE_DIRECTORY_ENV)
        if configured:
            return Path(configured).expanduser()

        xdg_cache = os.environ.get('XDG_CACHE_HOME')
        if xdg_cache:
            cache_root = Path(xdg_cache).expanduser()
        elif sys.platform == 'darwin':
            cache_root = Path.home() / 'Library' / 'Caches'
        elif (
                os.name == 'nt'
                and os.environ.get('LOCALAPPDATA')
        ):
            cache_root = Path(os.environ['LOCALAPPDATA'])
        else:
            cache_root = Path.home() / '.cache'
        return cache_root / 'tile2net'

    @property
    def path(self) -> Path:
        """Path to the per-user sqlite cache db, honoring the override env var."""
        configured = os.environ.get(SQLITE_PATH_ENV)
        if self._descriptor is not self:
            return self._descriptor.path
        if configured:
            path = Path(configured).expanduser()
        else:
            path = self.cache_directory() / 'nominatim.sqlite'
        self._path = path
        return path

    def __init__(
            self,
            func=None,
            read=True,
            write=True,
    ):
        """Store the read/write toggles for the sqlite cache layer."""
        super().__init__(func, read, write)

    def _key(
            self,
            action: str,
            query: Any,
            params: dict,
    ) -> str:
        """Hash the canonical representation into a fixed-length key."""
        canonical = super()._key(action, query, params)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the cache db, creating the table if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            'CREATE TABLE IF NOT EXISTS nominatim ('
            'key TEXT PRIMARY KEY, raw TEXT, accessed_at INTEGER)'
        )
        return conn

    def _read(self, key: str) -> Optional[dict]:
        """Fetch a cached raw response by key, bumping its recency, or None if absent."""
        with self._lock, closing(self._connect) as conn:
            row = conn.execute(
                'SELECT raw FROM nominatim WHERE key = ?', (key,)
            ).fetchone()
            if row is not None:
                conn.execute(
                    'UPDATE nominatim SET accessed_at = ? WHERE key = ?',
                    (time.time_ns(), key),
                )
        if row is None:
            return None
        return json.loads(row[0])

    def _write(
            self,
            key: str,
            raw: dict,
    ) -> None:
        """Upsert raw under key, then evict down to the `max_entries` most recent rows."""
        with self._lock, closing(self._connect) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO nominatim (key, raw, accessed_at) '
                'VALUES (?, ?, ?)',
                (key, json.dumps(raw), time.time_ns()),
            )
            conn.execute(
                'DELETE FROM nominatim WHERE key NOT IN ('
                'SELECT key FROM nominatim ORDER BY accessed_at DESC LIMIT ?'
                ')',
                (self.max_entries,),
            )


class Nominatim(geopy.geocoders.Nominatim):
    """Extended Nominatim geocoder, backed by json/sqlite caches."""

    @JSON
    def json(self):
        """
        Namespace for the JSON cache.

        >>> JSON._get_cache

        Examples:
            >>> geocoder = Nominatim(user_agent="tile2net")
            >>> geocoder.json.geocode("Berkeley, CA")
            >>> geocoder.json.reverse((37.8715, -122.2730))
        """

    @SQLite
    def sqlite(self):
        """
        Namespace for the SQLite cache.

        >>> SQLite._get_cache

        Examples:
            >>> geocoder = Nominatim(user_agent="tile2net")
            >>> geocoder.sqlite.geocode("Berkeley, CA")
            >>> geocoder.sqlite.reverse((37.8715, -122.2730))
        """

    def geocode(
            self,
            query,
            *,
            exactly_one=True,
            timeout=DEFAULT_SENTINEL,
            limit=None,
            addressdetails=False,
            language=False,
            geometry=None,
            extratags=False,
            country_codes=None,
            viewbox=None,
            bounded=False,
            featuretype=None,
            namedetails=False
    ):
        """
        Cache-aware wrapper around Nominatim.geocode, checked json then sqlite.

        query:
            The address, query, or a structured query you wish to geocode.
        exactly_one:
            If True, return a single Location object. If False, return a list of Location objects.
        timeout:
            The maximum number of seconds to wait for a response from the Nominatim service.
        limit:
            The maximum number of results to return from Nominatim. Unless exactly_one is
            False, limit will always be 1.
        addressdetails:
            If True, include structured address details in the Location.raw attribute.
        language:
            The preferred language in which to return results. This can be a string or a list of strings.
        geometry:
            If present, specifies whether to include the result's geometry, as
            one of `wkt`, `svg`, `kml`, or `geojson`.
        extratags:
            If True, include additional result information if available, e.g.
            wikipedia link, opening hours.
        country_codes:
            Limit search results to a specific country, or a list of countries,
            given as ISO 3166-1alpha2 codes.
        viewbox:
            Prefer this area to find search results. By default this is treated
            as a hint; combine with bounded=True to restrict results to it.
        bounded:
            If True, restrict results to only items contained within `viewbox`.
        featuretype:
            If present, restrict results to a certain feature type, e.g. `country`,
            `state`, `city`, `settlement`.
        namedetails:
            If True, include alternative namedetails in the Location.raw attribute.
        """
        params = dict(
            limit=limit,
            addressdetails=addressdetails,
            language=language,
            geometry=geometry,
            extratags=extratags,
            country_codes=country_codes,
            viewbox=viewbox,
            bounded=bounded,
            featuretype=featuretype,
            namedetails=namedetails,
        )

        def fetch_network():
            return super(Nominatim, self).geocode(
                query, exactly_one=exactly_one, timeout=timeout, **params
            )

        def fetch_sqlite():
            return self.sqlite.geocode(query, params, exactly_one, fetch_network)

        return self.json.geocode(query, params, exactly_one, fetch_sqlite)

    def reverse(
            self,
            query,
            *,
            exactly_one=True,
            timeout=DEFAULT_SENTINEL,
            language=False,
            addressdetails=True,
            zoom=None,
            namedetails=False,
    ):
        """
        Cache-aware wrapper around Nominatim.reverse, checked json then sqlite.

        query:
            The coordinates for which you wish to obtain the closest address.
            This can be a string, a tuple of (latitude, longitude), or a Point object.
        exactly_one:
            If True, return a single Location object. If False, return a list of Location objects
        timeout:
            The maximum number of seconds to wait for a response from the Nominatim service.
        language:
            The preferred language in which to return results. This can be a string or a list of strings.
        addressdetails:
            If True, include structured address details in the Location.raw attribute.
        zoom:
            The level of detail required for the address, from 0 (country) to 18
            (building). If None, the default zoom level will be used.
        namedetails:
            If True, include alternative namedetails in the Location.raw attribute.
        """
        params = dict(
            language=language,
            addressdetails=addressdetails,
            zoom=zoom,
            namedetails=namedetails,
        )

        def fetch_network():
            return super(Nominatim, self).reverse(
                query, exactly_one=exactly_one, timeout=timeout, **params
            )

        def fetch_sqlite():
            return self.sqlite.reverse(query, params, exactly_one, fetch_network)

        return self.json.reverse(query, params, exactly_one, fetch_sqlite)
