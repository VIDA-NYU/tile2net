from __future__ import annotations

from tile2net.core.source import Remote
from .dir import Dir


class SourceDir(
    Dir
):
    @Dir
    def static(self):
        ...

    @property
    def sample(self) -> str:
        """File path to a sample tile provided by the source."""
        dir = self.dir
        source = self.basegrid.source
        if not isinstance(source, Remote):
            msg = 'Sample tiles are only available for remote sources.'
            raise TypeError(msg)

