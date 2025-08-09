if False:
    from .grid import  Grid
class Loader(

):
    """Loader"""
    grid: Grid

    def __get__(
            self,
            instance: Grid,
            owner: type[Grid]
    ):
        self.grid = instance
        self.Grid = owner


