from __future__ import annotations

if False:
    # from tile2net.tiles.segtiles import padded
    from . import padded
    from . import broadcast

"""
Padded inherits from SegTiles, but we want define `SegTiles.padded` 
(causes circular dependency)
This allows us to swap-in the real subclass after SegTiles has been defined.
"""


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
        padded.__set_name__(owner, self.__name__)
        if instance is None:
            result = getattr(owner, self.__name__)
        else:
            result = getattr(instance, self.__name__)
        return result

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(
            self,
            *args,
    ):
        ...


class Broadcast(

):

    def __get__(
            self,
            instance,
            owner
    ) -> broadcast.Broadcast:
        from . import broadcast
        broadcast = broadcast.Broadcast()
        setattr(owner, self.__name__, broadcast)
        broadcast.__set_name__(owner, self.__name__)
        if instance is None:
            result = getattr(owner, self.__name__)
        else:
            result = getattr(instance, self.__name__)
        return result

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__( self, *args, ):
        ...
