if False:
    from .grid import  Tiles
class Loader(

):
    """Loader"""
    grid: Tiles

    def __get__(
            self,
            instance: Tiles,
            owner: type[Tiles]
    ):
        self.grid = instance
        self.Tiles = owner


