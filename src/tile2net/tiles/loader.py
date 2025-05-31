if False:
    from .tiles import  Tiles
class Loader(

):
    """Loader"""
    tiles: Tiles

    def __get__(
            self,
            instance: Tiles,
            owner: type[Tiles]
    ):
        self.tiles = instance
        self.Tiles = owner


