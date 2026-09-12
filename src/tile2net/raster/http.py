from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SENSITIVE_QUERY_PARAMETERS = frozenset(
    {
        "api_key",
        "access_token",
        "key",
        "signature",
        "token",
    }
)


def redact_url(url: str) -> str:
    """Return a URL with security-sensitive query values removed."""
    parts = urlsplit(url)
    query = [
        (
            name,
            "redacted" if name.casefold() in SENSITIVE_QUERY_PARAMETERS else value,
        )
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(parts._replace(query=urlencode(query)))
