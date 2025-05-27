
"""
tiles.file
mosaic.file
"""
import pandas as pd

if False:
    from .tiles import  Tiles

class Mosaic(

):

    def __get__(
            self,
            instance,
            owner
    ):
        ...

    @property
    def tx(self) -> pd.Series[int]:
        """Tile integer X of this tile in the stitched mosaic"""

    @property
    def ty(self) -> pd.Series[int]:
        """Tile integer Y of this tile in the stitched mosaic"""

    @property
    def r(self):
        """row within the mosaic of this tile"""

    @property
    def c(self):
        """column within the mosaic of this tile"""

    @property
    def px(self):
        """Starting pixel X coordinate of this tile in the stitched mosaic"""

    @property
    def py(self):
        """Starting pixel Y coordinate of this tile in the stitched mosaic"""










