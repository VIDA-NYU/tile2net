from __future__ import annotations
from ..grid import padded

from .ingrid import InGrid


def __get__(
        self: Padded,
        instance: InGrid,
        owner,
) -> Padded:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        seggrid = instance.seggrid
        result = (
            instance
            .to_scale(seggrid.scale)
            .to_padding()
            .to_scale(instance.scale)
            .pipe(self.__class__)
        )
        result.__dict__.update(instance.__dict__)
        result.instance = instance
        instance.__dict__[self.__name__] = result
    return result


class Padded(
    padded.Padded,
    InGrid,
):
    locals().update(
        __get__=__get__
    )
