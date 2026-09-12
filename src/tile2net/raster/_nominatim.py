from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from geopy.geocoders import Nominatim as _Nominatim
from geopy.location import Location

CACHE_DIRECTORY_ENV = 'TILE2NET_CACHE_DIR'
SQLITE_PATH_ENV = 'TILE2NET_NOMINATIM_CACHE'


def cache_directory() -> Path:
    """User-writable cache directory, mirroring raster.weights.weights_directory."""
    configured = os.environ.get(CACHE_DIRECTORY_ENV)
    if configured:
        return Path(configured).expanduser()

    xdg_cache = os.environ.get('XDG_CACHE_HOME')
    if xdg_cache:
        cache_root = Path(xdg_cache).expanduser()
    elif sys.platform == 'darwin':
        cache_root = Path.home() / 'Library' / 'Caches'
    # elif os.name == 'nt' and os.environ.get('LOCALAPPDATA'):
    elif (
        os.name == 'nt'
        and os.environ.get('LOCALAPPDATA')
    ):
        cache_root = Path(os.environ['LOCALAPPDATA'])
    else:
        cache_root = Path.home() / '.cache'
    return cache_root / 'tile2net'

class Nominatim(_Nominatim):
    """
    Extends geopy's Nominatim geocoder with a lookup cache consulted
    before hitting the Nominatim servers: a repo-tracked JSON file of
    known test locations, and a per-user SQLite database populated as
    new locations are queried. See github.com/VIDA-NYU/tile2net#92.
    """

    _use_json: bool = False
    _use_sqlite: bool = True
    _json_path: Path = Path(__file__, '../', 'nominatim.json').resolve()
    _json_fields = 'display_name', 'boundingbox'
    _lock = threading.Lock()
    _json_cache: Optional[dict[str, dict]] = None

    @classmethod
    def _sqlite_path(cls) -> Path:
        """Resolve the per-user sqlite cache path, honoring the override env var."""
        configured = os.environ.get(SQLITE_PATH_ENV)
        if configured:
            return Path(configured).expanduser()
        return cache_directory() / 'nominatim.sqlite'

    @classmethod
    def _load_json(cls) -> dict[str, dict]:
        """Load and memoize the tracked json cache from disk."""
        if cls._json_cache is None:
            if cls._json_path.exists():
                with open(cls._json_path) as f:
                    cls._json_cache = json.load(f)
            else:
                cls._json_cache = {}
        return cls._json_cache

    @classmethod
    def _dump_json(cls) -> None:
        """Persist the in-memory json cache back to disk."""
        cls._json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cls._json_path, 'w') as f:
            json.dump(cls._json_cache, f, indent=2, sort_keys=True)
            f.write('\n')

    @classmethod
    def _connect(cls) -> sqlite3.Connection:
        """Open a sqlite connection to the cache db, creating the table if needed."""
        path = cls._sqlite_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        sql = 'CREATE TABLE IF NOT EXISTS nominatim (key TEXT PRIMARY KEY, raw TEXT)'
        conn.execute(sql)
        return conn

    @classmethod
    def _sqlite_get(cls, key: str) -> Optional[dict]:
        """Fetch a cached raw response by key, or None if absent."""
        with cls._lock, cls._connect() as conn:
            sql = 'SELECT raw FROM nominatim WHERE key = ?'
            row = (
                conn.execute(sql, (key,))
                .fetchone()
            )
        if row is None:
            return None
        return json.loads(row[0])

    @classmethod
    def _sqlite_set(cls, key: str, raw: dict) -> None:
        """Upsert a raw response into the sqlite cache."""
        parameters = key, json.dumps(raw)
        with cls._lock, cls._connect() as conn:
            sql = 'INSERT OR REPLACE INTO nominatim (key, raw) VALUES (?, ?)'
            conn.execute(sql, parameters)

    @staticmethod
    def _normalize(query: Any) -> str:
        """Stringify a geocode/reverse query into a stable cache key."""
        if hasattr(query, 'latitude') and hasattr(query, 'longitude'):
            return f'{query.latitude},{query.longitude}'
        if isinstance(query, (tuple, list)):
            return ','.join(str(value) for value in query)
        return str(query)

    @staticmethod
    def _to_location(raw: dict) -> Location:
        """Reconstruct a geopy Location from a cached raw response."""
        address = raw.get('display_name', '')
        lat = float(raw.get('lat', 0.0))
        lon = float(raw.get('lon', 0.0))
        return Location(address, (lat, lon), raw)

    @classmethod
    def _to_json_value(cls, raw: dict) -> dict:
        """Trim a raw response down to the fields tracked in json."""
        return {
            field: raw[field]
            for field in cls._json_fields
            if field in raw
        }

    def _cached(
            self,
            query: Any,
            exactly_one: bool,
            wrapped,
    ) -> Optional[Location]:
        """
        Check json/sqlite caches before falling back to wrapped, caching new results.

        query:
            The geocode/reverse query to normalize into a cache key.
        exactly_one:
            Return one result or a list of results, if available.
        """

        if not exactly_one:
            # list results aren't cached; this codebase only ever
            # requests exactly_one geocodes/reverses
            return wrapped()

        key = self._normalize(query)
        # json ships with the library as a fixed set of known
        # locations for quick testing, so it's checked before the
        # per-user, dynamically populated sqlite cache
        if self._use_json:
            raw = self._load_json().get(key)
            if raw is not None:
                return self._to_location(raw)

        if self._use_sqlite:
            raw = self._sqlite_get(key)
            if raw is not None:
                return self._to_location(raw)

        result = wrapped()
        if result is not None:
            if self._use_sqlite:
                self._sqlite_set(key, result.raw)
            if self._use_json:
                self._load_json()[key] = self._to_json_value(result.raw)
                self._dump_json()
        return result

    def geocode(self, query, *, exactly_one=True, **kwargs):
        """Cache-aware wrapper around Nominatim.geocode."""
        def wrapped():
            return super(Nominatim, self).geocode( query, exactly_one=exactly_one, **kwargs)
        return self._cached(query, exactly_one, wrapped)

    def reverse(self, query, *, exactly_one=True, **kwargs):
        """Cache-aware wrapper around Nominatim.reverse."""
        def wrapped():
            return super(Nominatim, self).reverse( query, exactly_one=exactly_one, **kwargs)
        return self._cached(query, exactly_one, wrapped)
