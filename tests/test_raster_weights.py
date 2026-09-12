from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tile2net.raster.project import Project
from tile2net.raster.weights import (
    WEIGHT_SPECS,
    WeightSpec,
    WeightValidationError,
    download_weight,
    validate_weight,
    weights_directory,
)


class FakeResponse:
    """Minimal streamed response used without network access."""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self.content[index:index + chunk_size]
            for index in range(0, len(self.content), chunk_size)
        ]


class FakeSession:
    """Record downloader calls and return deterministic checkpoint bytes."""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.urls: list[str] = []

    def get(
        self,
        url: str,
        *,
        stream: bool,
        timeout: tuple[int, int],
    ) -> FakeResponse:
        assert stream is True
        assert timeout == (15, 300)
        self.urls.append(url)
        return FakeResponse(self.content)


def make_spec(content: bytes) -> WeightSpec:
    """Create a small immutable manifest entry for offline tests."""
    return WeightSpec(
        filename="checkpoint.pth",
        doi="10.6084/m9.figshare.synthetic.v1",
        download_url="https://example.test/checkpoint.pth",
        size_bytes=len(content),
        md5=hashlib.md5(content, usedforsecurity=False).hexdigest(),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_published_manifest_is_complete_and_immutable() -> None:
    assert len(WEIGHT_SPECS) == 2
    assert len({spec.filename for spec in WEIGHT_SPECS}) == 2
    for spec in WEIGHT_SPECS:
        assert spec.doi.endswith(".v1")
        assert spec.download_url.startswith(
            "https://ndownloader.figshare.com/files/"
        )
        assert spec.size_bytes > 0
        assert len(spec.md5) == 32
        assert len(spec.sha256) == 64


def test_weights_directory_uses_hpc_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TILE2NET_WEIGHTS_DIR", os.fspath(tmp_path))

    assert weights_directory() == tmp_path
    assert Project.resources.assets.weights.path == tmp_path


def test_download_weight_streams_validates_and_installs_atomically(
    tmp_path: Path,
) -> None:
    content = b"synthetic checkpoint bytes"
    spec = make_spec(content)
    session = FakeSession(content)

    destination = download_weight(spec, tmp_path, session=session)

    assert destination.read_bytes() == content
    assert session.urls == [spec.download_url]
    assert not tuple(tmp_path.glob("*.part"))
    validate_weight(destination, spec)


def test_download_weight_reuses_verified_cache(tmp_path: Path) -> None:
    content = b"existing checkpoint"
    spec = make_spec(content)
    destination = tmp_path / spec.filename
    destination.write_bytes(content)
    session = FakeSession(b"must not be downloaded")

    result = download_weight(spec, tmp_path, session=session)

    assert result == destination
    assert session.urls == []


def test_failed_download_preserves_existing_checkpoint(tmp_path: Path) -> None:
    expected = b"expected checkpoint"
    existing = b"existing invalid checkpoint"
    spec = make_spec(expected)
    destination = tmp_path / spec.filename
    destination.write_bytes(existing)

    with pytest.raises(WeightValidationError, match="Unexpected Content-Length"):
        download_weight(spec, tmp_path, session=FakeSession(b"wrong"))

    assert destination.read_bytes() == existing
    assert not tuple(tmp_path.glob("*.part"))


def test_download_aborts_when_stream_exceeds_pinned_size(
    tmp_path: Path,
) -> None:
    content = b"expected"
    spec = make_spec(content)
    response_content = content + b"unexpected trailing bytes"
    session = FakeSession(response_content)
    original_get = session.get

    def get_without_length(*args: object, **kwargs: object) -> FakeResponse:
        response = original_get(*args, **kwargs)
        response.headers = {}
        return response

    session.get = get_without_length  # type: ignore[method-assign]

    with pytest.raises(WeightValidationError, match="exceeded the pinned size"):
        download_weight(spec, tmp_path, session=session)

    assert not (tmp_path / spec.filename).exists()
    assert not tuple(tmp_path.glob("*.part"))
