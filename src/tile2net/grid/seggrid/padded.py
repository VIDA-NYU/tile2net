from __future__ import annotations

from ...grid.frame.namespace import namespace

if False:
    from .seggrid import SegGrid


class Padded(
    namespace
):

    instance: SegGrid
    @property
    def length(self) -> int:
        result = self.instance.length + 2
        return result

    @property
    def dimension(self) -> int:
        return self.instance.ingrid.dimension * self.length





