from __future__ import annotations

from .dir import Dir


class SourceDir(
    Dir
):
    @Dir
    def static(self):
        ...
