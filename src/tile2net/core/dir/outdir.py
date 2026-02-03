from __future__ import annotations

import copy
import os
import os.path
from typing import *

from tile2net.core.dir import projectdir
from tile2net.core.dir.dir import Dir
from tile2net.core.dir.exceptions import XYNotFoundError
from tile2net.core.dir.projectdir import ProjectDir
from tile2net.core.dir.sourcedir import SourceDir
from tile2net.core.source.remote import Remote

if TYPE_CHECKING:
    from .seggrid import SegGrid
    from .vecgrid import VecGrid
    from tile2net.core.grid import Grid


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
        from tile2net.core.grid import Grid
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
        itile = (
            instance.cfg.itile
            or instance.ITILE
        )

        if isinstance(value, str):
            value = self.from_template(value, itile=itile)
        elif isinstance(value, Dir):
            ...
        else:
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

    @Dir
    def static(self):
        ...

    @property
    def network(self) -> projectdir.Network:
        return self.project.network

    @property
    def polygons(self) -> projectdir.Polygons:
        return self.project.polygons

    @property
    def vecgrid(self) -> projectdir.VecGrid:
        return self.project.vecgrid

    @property
    def seggrid(self) -> projectdir.SegGrid:
        return self.project.seggrid

    @property
    def grid(self) -> projectdir.Grid:
        return self.project.grid

    @ProjectDir
    def project(self):
        """Subdirectory named after the particular directory.
        Uses the passed name, location, or bounding box.
        """
        grid = self.basegrid
        ymin, xmin, ymax, xmax = grid.lat_lon
        name = (
                grid.cfg.name
                or grid.location
                or f"{ymin:.2f},{xmin:.2f},{ymax:.2f},{xmax:.2f}"
        )
        result = ProjectDir.from_parent(self, name)
        return result

    @SourceDir
    def source(self):
        source = self.basegrid.source
        if isinstance(source, Remote):
            name = source.name
            return SourceDir.from_parent(self, name)
        else:
            msg = 'Outdir.source is only available for Remote sources.'
            raise TypeError(msg)
