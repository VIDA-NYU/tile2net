from __future__ import annotations

from typing import *

from tile2net.eval.file import File
from tile2net.core.frame.namespace import namespace

if TYPE_CHECKING:
    from .eval import Eval

class GridNamespace(
    namespace
):
    wrapper: Eval

    @File
    def file(self):
        ...

