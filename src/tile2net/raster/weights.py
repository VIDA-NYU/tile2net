from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tile2net.logger import logger

WEIGHTS_DIRECTORY_ENV = "TILE2NET_WEIGHTS_DIR"
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """Immutable identity and download metadata for one model checkpoint."""

    filename: str
    doi: str
    download_url: str
    size_bytes: int
    md5: str
    sha256: str


WEIGHT_SPECS: tuple[WeightSpec, ...] = (
    WeightSpec(
        filename="satellite_2021.pth",
        doi="10.6084/m9.figshare.33315570.v1",
        download_url="https://ndownloader.figshare.com/files/67765602",
        size_bytes=578_433_769,
        md5="6718e1b06c57cb884dbe93e7d71f446a",
        sha256="745f8c099e98f112a152aedba493f61fb6d80c1761e5866f936eb5f361c7ab4d",
    ),
    WeightSpec(
        filename="hrnetv2_w48_imagenet_pretrained.pth",
        doi="10.6084/m9.figshare.33315558.v1",
        download_url="https://ndownloader.figshare.com/files/67765578",
        size_bytes=310_643_500,
        md5="1ce00d88068e5f483de64ea48693bc9a",
        sha256="0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8",
    ),
)


class WeightValidationError(RuntimeError):
    """Raised when a checkpoint does not match its pinned manifest entry."""


def weights_directory() -> Path:
    """Return the configurable, user-writable checkpoint cache directory."""
    configured = os.environ.get(WEIGHTS_DIRECTORY_ENV)
    if configured:
        return Path(configured).expanduser()

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        cache_root = Path(xdg_cache).expanduser()
    elif sys.platform == "darwin":
        cache_root = Path.home() / "Library" / "Caches"
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        cache_root = Path(os.environ["LOCALAPPDATA"])
    else:
        cache_root = Path.home() / ".cache"
    return cache_root / "tile2net" / "weights"


def _digest_file(path: Path) -> tuple[str, str]:
    """Calculate SHA-256 and MD5 in one sequential read."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(DOWNLOAD_CHUNK_SIZE), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def validate_weight(path: Path, spec: WeightSpec) -> None:
    """Validate one checkpoint against its pinned size and cryptographic hashes."""
    if not path.is_file():
        raise WeightValidationError(f"Checkpoint is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size_bytes:
        raise WeightValidationError(
            f"Invalid size for {spec.filename}: expected {spec.size_bytes:,} "
            f"bytes, found {actual_size:,}."
        )

    actual_sha256, actual_md5 = _digest_file(path)
    if actual_sha256 != spec.sha256 or actual_md5 != spec.md5:
        raise WeightValidationError(
            f"Integrity check failed for {spec.filename}. "
            "Delete the invalid cache file and run generate again."
        )


def _download_session() -> requests.Session:
    """Create an HTTP session with bounded retries for immutable public files."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_weight(
    spec: WeightSpec,
    directory: Path,
    *,
    session: requests.Session | None = None,
) -> Path:
    """Stream, validate, and atomically install one checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / spec.filename
    if destination.exists():
        try:
            validate_weight(destination, spec)
        except WeightValidationError:
            logger.warning("Replacing invalid cached checkpoint %s.", destination)
        else:
            return destination

    owned_session = session is None
    http = session or _download_session()
    temporary_path: Path | None = None
    try:
        logger.info(
            "Downloading %s (%.1f MiB)...",
            spec.filename,
            spec.size_bytes / 2**20,
        )
        with http.get(
            spec.download_url,
            stream=True,
            timeout=(15, 300),
        ) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) != spec.size_bytes:
                raise WeightValidationError(
                    f"Unexpected Content-Length for {spec.filename}: "
                    f"expected {spec.size_bytes:,}, received {int(content_length):,}."
                )

            with tempfile.NamedTemporaryFile(
                dir=directory,
                prefix=f".{spec.filename}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                downloaded = 0
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        downloaded += len(chunk)
                        if downloaded > spec.size_bytes:
                            raise WeightValidationError(
                                f"Download exceeded the pinned size for "
                                f"{spec.filename}: expected "
                                f"{spec.size_bytes:,} bytes."
                            )
                        temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())

        validate_weight(temporary_path, spec)
        temporary_path.replace(destination)
        temporary_path = None
        logger.info("Installed verified checkpoint at %s.", destination)
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if owned_session:
            http.close()


def ensure_weights(
    directory: Path | None = None,
    *,
    session: requests.Session | None = None,
) -> dict[str, Path]:
    """Ensure every pinned Tile2Net checkpoint is present and valid."""
    target = weights_directory() if directory is None else Path(directory)
    owned_session = session is None
    http = session or _download_session()
    try:
        return {
            spec.filename: download_weight(spec, target, session=http)
            for spec in WEIGHT_SPECS
        }
    finally:
        if owned_session:
            http.close()
