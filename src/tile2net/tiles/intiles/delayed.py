from __future__ import annotations

if False:
    from . import padded

"""
Filled inherits from SegTiles, but we want define `SegTiles.padded` 
(causes circular dependency)
This allows us to swap-in the real subclass after SegTiles has been defined.
"""


class Filled(

):
    def __get__(
            self,
            instance,
            owner
    ) -> padded.Filled:
        from . import padded
        padded = padded.Filled()
        setattr(owner, self.__name__, padded)
        padded.__set_name__(owner, self.__name__)
        if instance is None:
            result = getattr(owner, self.__name__)
        else:
            result = getattr(instance, self.__name__)
        return result

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__( self, *args, ):
        ...
