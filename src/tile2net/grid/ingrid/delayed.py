from __future__ import annotations

if False:
    from . import padded

"""
Padded inherits from SegGrid, but we want define `SegGrid.padded` 
(causes circular dependency)
This allows us to swap-in the real subclass after SegGrid has been defined.
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

    def __init__( self, *args, ):
        ...
