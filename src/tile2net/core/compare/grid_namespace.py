from __future__ import annotations

from typing import *

from tile2net.core.compare.file import File
from tile2net.core.frame.namespace import namespace

if TYPE_CHECKING:
    from .compare import Compare

class GridNamespace(
    namespace
):
    wrapper: Compare

    @File
    def file(self):
        ...

