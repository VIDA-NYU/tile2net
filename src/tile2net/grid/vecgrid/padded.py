from __future__ import annotations

from ...grid.frame.namespace import namespace

if False:
    from .vecgrid import VecGrid


class Padded(
    namespace
):

    instance: VecGrid
    @property
    def length(self) -> int:
        result = self.instance.length + 2
        return result

    @property
    def dimension(self) -> int:
        return self.instance.seggrid.dimension * self.length





