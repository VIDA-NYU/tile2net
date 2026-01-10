from __future__ import annotations

import os
import os.path

from .dir import Dir
from .sourcedir import SourceDir

if False:
    import tile2net.grid.ingrid
    from .seggrid import SegGrid
    from .vecgrid import VecGrid


class Probability(
    Dir,
):
    extension = 'tif'


class Prediction(
    Dir
):
    extension = 'png'


class Error(
    Dir
):
    extension = 'npy'


class SideBySide(
    Dir
):
    ...


class Submit(
    Dir,
):
    ...


class MaskRaw(
    Dir,
):
    ...


class Mask(
    Dir,
):
    ...


class BestImages(
    Dir
):
    @property
    def webpage(self):
        return os.path.join(self.dir, 'webpage.html')


class Outdir(
    Dir
):
    grid: tile2net.grid.ingrid.InGrid

    @SourceDir
    def sourcedir(self):
        """
        Handles lazy-loading of sourcedir:
        >>> SourceDir._get
        """
        grid = self.grid
        name = None
        try:
            name = grid.source.name
        except ValueError:
            ...
        if name is None:
            name = grid.cfg.indir.name
        if name is None:
            name = grid.indir.dir.rsplit(os.sep, 1)[-1]
        format = os.path.join(
            self.dir,
            name,
            self.suffix
        )
        result = SourceDir.from_format(format)
        return result

    @property
    def network(self):
        return self.sourcedir.network

    @property
    def polygons(self):
        return self.sourcedir.polygons

    @property
    def vecgrid(self) -> VecGrid:
        return self.sourcedir.namedir.vecgrid

    @property
    def seggrid(self) -> SegGrid:
        return self.sourcedir.namedir.seggrid

    @property
    def namedir(self):
        return self.sourcedir.namedir

    @property
    def ingrid(self):
        return self.sourcedir.namedir.ingrid
