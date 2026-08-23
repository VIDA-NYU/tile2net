from __future__ import annotations

import os
from collections.abc import Iterable
from os import PathLike


def filter_paths_by_tile_ids(
    paths: Iterable[str | PathLike[str]],
    active_tile_ids: Iterable[int] | None,
) -> list[str | PathLike[str]]:
    """Filter stitched-image paths using their final ``_<tile_id>`` token."""
    if active_tile_ids is None:
        return list(paths)

    active_ids = frozenset(map(int, active_tile_ids))
    selected: list[str | PathLike[str]] = []
    for path in paths:
        filename = os.path.basename(os.fspath(path))
        stem = filename.rsplit(".", maxsplit=1)[0]
        _, separator, tile_id = stem.rpartition("_")
        if not separator:
            continue
        try:
            parsed_id = int(tile_id)
        except ValueError:
            continue
        if parsed_id in active_ids:
            selected.append(path)
    return selected
