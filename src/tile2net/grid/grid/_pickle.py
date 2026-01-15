import tempfile

from tile2net.grid.cfg.logger import logger
from tile2net.grid.frame.namespace import namespace
from pathlib import Path
import pooch

from .grid import Grid


class VRAM24(
    namespace
):
    def from_boston_common(
            self,
            outdir='./outdir',
            location='Boston Common',
            pad=2,
            length=8,
            force=False,
    ) -> Grid:
        path = Path(
            tempfile.gettempdir(),
            'tile2net',
            'VRAM24',
            'boston_common.pkl',
        )
        if (
                not path.exists()
                or force
        ):
            result = Grid.construct.vram24.from_boston_common(
                outdir=outdir,
                location=location,
                pad=pad,
                length=length,
            )
            result.to_pickle(path)
        else:
            result = Grid.from_pickle(path)
        return result


    def from_manhattan(self) -> Grid:
        ...

    def from_portland(self) -> Grid:
        ...

    # def to_boston_common(
    #         self,
    #         outdir='./outdir',
    #         location='Boston Common',
    #         pad=2,
    #         length=8,
    #         path='../../tile2net-pickle/VRAM24/boston_common.pkl',
    # ):
    #     """Small park in downtown Boston."""
    #     grid = Grid.construct.vram24.from_boston_common(
    #         outdir=outdir,
    #         location=location,
    #         pad=pad,
    #         length=length,
    #     )
    #     result = grid.to_pickle(path)
    #     print(result.path)
    #     print(result.md5)
    #     return result

    def to_dir(
            self,
            path,
            outdir=None,
            pad=None,
            length=None,
    ):
        path = Path(path) / 'VRAM24'
        grid = Grid.construct.vram24.from_boston_common(
            outdir=outdir,
            pad=pad,
            length=length,
        )
        p = path / 'boston_common.pkl'
        result = grid.to_pickle(p)
        print(result.path)
        print(result.md5)


class VRAM8(
    namespace
):
    ...


class Pickle(
    namespace
):

    @VRAM8
    def vram8(self):
        """Pickle constructors optimized for local inference with 8 GB VRAM."""

    @VRAM24
    def vram24(self):
        """Pickle constructors optimized for local inference with 24 GB VRAM."""


if __name__ == '__main__':
    ...

    # Grid.pickle.vram24.to_dir(
    #     '../../tile2net-pickle',
    # )
    # Grid
