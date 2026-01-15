from __future__ import annotations

import copy
import os
import os.path
from typing import *

from tile2net.grid.dir import namedir, sourcedir
from tile2net.grid.dir.dir import Dir
from tile2net.grid.dir.exceptions import XYNotFoundError
from tile2net.grid.dir.sourcedir import SourceDir
from tile2net.grid.source.remote import Remote

if TYPE_CHECKING:
    from .seggrid import SegGrid
    from .vecgrid import VecGrid
    from tile2net.grid.grid.grid import Grid


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
    def _get(
            self,
            instance: Grid,
            owner: type[Grid],
    ):
        from tile2net.grid.grid import Grid
        if instance is None:
            out = self
        elif isinstance(instance, Grid):
            cache = instance.__dict__
            name = self.__name__
            if name not in cache:
                self.__set__(instance, instance.cfg.outdir)
            out = cache[name]
        else:
            raise TypeError(instance)

        out.basegrid = instance
        return out

    locals().update(__get__=_get)

    def __set__(
            self,
            instance: Grid,
            value: str | Dir,
    ):
        if isinstance(value, str):
            try:
                value = self.from_template(value)
            except XYNotFoundError:
                item = os.path.join(value, instance.cfg.template)
                value = self.from_template(item)
        if not isinstance(value, Dir):
            raise TypeError(value)
        instance.__dict__[self.__name__] = copy.copy(value)

    def __delete__(
            self,
            instance: Grid,
    ):
        """"""
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            ...

    @property
    def network(self) -> sourcedir.Network:
        return self.sourcedir.network

    @property
    def polygons(self) -> sourcedir.Polygons:
        return self.sourcedir.polygons

    @SourceDir
    def sourcedir(self):
        """Subdirectory named after the source. This si th eparent fo the project directory,
        so that downloaded images may be reused across multiple projects. """
        grid = self.basegrid
        name = None
        if isinstance(grid.source, Remote):
            name = grid.source.name
        if name is None:
            name = grid.cfg.indir.name
        if name is None:
            name = grid.indir.dir.rsplit(os.sep, 1)[-1]
        template = os.path.join(self.dir, name, self.suffix)
        result = SourceDir.from_template(template)
        return result

    @property
    def vecgrid(self) -> namedir.VecGrid:
        return self.sourcedir.namedir.vecgrid

    @property
    def seggrid(self) -> namedir.SegGrid:
        return self.sourcedir.namedir.seggrid

    @property
    def grid(self) -> namedir.Grid:
        return self.sourcedir.namedir.grid

    @property
    def namedir(self) -> namedir.NameDir:
        return self.sourcedir.namedir
