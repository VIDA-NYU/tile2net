from __future__ import annotations
from ..tiles import padded

from .intiles import InTiles


def __get__(
        self: Filled,
        instance: InTiles,
        owner,
) -> Filled:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        segtiles = instance.segtiles
        result = (
            instance
            .to_scale(segtiles.tile.scale)
            .to_padding()
            .to_scale(instance.tile.scale)
            .pipe(self.__class__)
        )
        result.__dict__.update(instance.__dict__)
        result.instance = instance
        instance.__dict__[self.__name__] = result
    return result


class Filled(
    padded.Filled,
    InTiles,
):
    locals().update(
        __get__=__get__
    )
