from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

import tile2net.raster.source as source_module
from tile2net.raster.http import redact_url
from tile2net.raster.source import Maine, VexCel


def test_maine_key_is_resolved_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(Maine.api_key_env, raising=False)

    with pytest.raises(RuntimeError, match=Maine.api_key_env):
        Maine._get_api_key()

    monkeypatch.setenv(Maine.api_key_env, "synthetic-test-key")
    assert Maine._get_api_key() == "synthetic-test-key"


def test_source_module_contains_no_vexcel_key_literal() -> None:
    source_text = Path(source_module.__file__).read_text(encoding="utf-8")

    assert "vfa_" not in source_text


def test_redact_url_preserves_non_sensitive_parameters() -> None:
    url = "https://example.test/tile?api_key=synthetic-secret&tile-x=12"
    redacted = redact_url(url)
    query = parse_qs(urlsplit(redacted).query)

    assert "synthetic-secret" not in redacted
    assert query == {"api_key": ["redacted"], "tile-x": ["12"]}


def test_vexcel_http_error_does_not_expose_key() -> None:
    response = requests.Response()
    response.status_code = 401
    response.reason = "Unauthorized"
    response.headers["Content-Type"] = "text/plain"
    response._content = b"access denied: synthetic-secret"
    url = "https://example.test/tile?api_key=synthetic-secret&tile-x=12"

    with pytest.raises(requests.HTTPError) as error:
        VexCel._raise_http(response, url)

    message = str(error.value)
    assert "synthetic-secret" not in message
    assert "api_key=redacted" in message
