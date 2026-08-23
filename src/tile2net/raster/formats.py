from __future__ import annotations

from enum import StrEnum


class VectorFormat(StrEnum):
    """Supported user-facing vector output formats."""

    PARQUET = "parquet"
    SHAPEFILE = "shapefile"

    @property
    def suffix(self) -> str:
        return ".parquet" if self is VectorFormat.PARQUET else ".shp"
