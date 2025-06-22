from __future__ import annotations
from ..tiles import padded

from .intiles import InTiles


def __get__(
        self: Padded,
        instance: InTiles,
        owner,
) -> Padded:
    if instance is None:
        result = self
    elif self.__name__ in instance.attrs:
        result = instance.attrs[self.__name__]
    else:
        segtiles = instance.segtiles
        result = (
            instance
            .to_scale(segtiles.tile.scale)
            .to_padding()
            .to_scale(instance.tile.scale)
            .pipe(self.__class__)
        )
        result.attrs.update(instance.attrs)
        result.instance = instance
        instance.attrs[self.__name__] = result
    return result


class Padded(
    padded.Padded,
    InTiles,
):
    locals().update(
        __get__=__get__
    )
