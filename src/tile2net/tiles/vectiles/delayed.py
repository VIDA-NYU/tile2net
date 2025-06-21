from __future__ import annotations

if False:
    from ..segtiles import padded


class Padded(

):
    def __get__(
            self,
            instance,
            owner
    ) -> padded.Padded:
        from . import padded
        padded = padded.Padded()
        setattr(owner, self.__name__, padded)
        result = getattr(owner, self.__name__)
        return result

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(
        self,
        *args,
    ):
        ...



