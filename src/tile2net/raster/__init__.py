from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ("Raster",)

if TYPE_CHECKING:
    from tile2net.raster.raster import Raster


def __getattr__(name: str) -> Any:
    """Preserve ``tile2net.raster.Raster`` without eager orchestration imports."""
    if name != "Raster":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from tile2net.raster.raster import Raster

    globals()[name] = Raster
    return Raster


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
