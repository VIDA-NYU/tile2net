from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ("Grid", "PedNet", "Project", "Raster", "Source", "Tile")

_EXPORTS = {
    "Grid": ("tile2net.raster.grid", "Grid"),
    "Raster": ("tile2net.raster.raster", "Raster"),
    "Tile": ("tile2net.raster.tile", "Tile"),
    "PedNet": ("tile2net.raster.pednet", "PedNet"),
    "Project": ("tile2net.raster.project", "Project"),
    "Source": ("tile2net.raster.source", "Source"),
    "logger": ("tile2net.logger", "logger"),
}


def __getattr__(name: str) -> Any:
    """Load public objects only when callers request them."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
