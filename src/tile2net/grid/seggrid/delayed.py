from __future__ import annotations

if False:
    # from tile2net.grid.seggrid import filled
    from . import filled
    from . import broadcast

class Filled:
    """
    Filled inherits from SegGrid, but we want to define `SegGrid.filled`
    (causes circular dependency)
    This allows us to swap-in the real subclass after SegGrid has been defined.
    """

    def __get__(
            self,
            instance,
            owner
    ) -> filled.Filled:
        from . import filled
        filled = filled.Filled()
        setattr(owner, self.__name__, filled)
        filled.__set_name__(owner, self.__name__)
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


class Broadcast:
    """
    Broadcast inherits from SegGrid, but we want to define `SegGrid.broadcast`
    (causes circular dependency)
    This allows us to swap-in the real subclass after SegGrid has been defined.
    """

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

    def __init__(self, *args, ):
        ...
