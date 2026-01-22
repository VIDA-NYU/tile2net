"""Exceptions for tile2net source parsing and construction."""

from __future__ import annotations


class SourceParseError(ValueError):
    """
    Base exception for source parsing errors.

    Raised when input doesn't match the expected format for a particular
    source construction method. This allows graceful fallback to alternative
    parsing strategies.
    """


class InvalidLocalPath(SourceParseError):
    """
    Raised when a string cannot be parsed as a Local source path.

    Indicates that the input is not a valid local filesystem path with
    the required {x}, {y}, {z} placeholders.
    """


class InvalidRemoteName(SourceParseError):
    """
    Raised when a string is not a registered Remote source name.

    Indicates that the input should be tried with other Remote construction
    methods like from_url or from_location.
    """


class InvalidRemoteUrl(SourceParseError):
    """
    Raised when a URL cannot be parsed as a Remote source.

    Indicates that the URL is malformed or missing required {x}, {y}, {z}
    placeholders for tile coordinates.
    """


class InvalidLocation(SourceParseError):
    """
    Raised when a location string cannot be geocoded or matched to coverage.

    Indicates that the input cannot be resolved to a geographic location
    or no Remote source covers the resolved location.
    """


class RemoteNotFound(ValueError):
    """
    Raised when no Remote source is found for a given location.

    Indicates that the geocoded location does not fall within the coverage
    of any available Remote tile sources.
    """