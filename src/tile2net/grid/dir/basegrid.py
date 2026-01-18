from __future__ import annotations

from tile2net.grid.dir.dir import Dir


class Postprocess(
    Dir
):
    basegrid: BaseGrid

    @Dir
    def pred(self):
        ...

    @Dir
    def colorized(self):
        ...

    @Dir
    def overlay(self):
        ...

    @Dir
    def prob(self):
        ...

    @Dir
    def error(self):
        ...

    @Dir
    def intensity(self):
        ...


class BaseGrid(
    Dir,
):
    padded: Padded

    @Dir
    def postprocess(self):
        ...

    @Dir
    def pred(self):
        ...

    @Dir
    def colorized(self):
        ...

    @Dir
    def static(self):
        ...

    @Dir
    def overlay(self):
        ...

    @Dir
    def prob(self):
        return Dir.from_parent(self, 'prob', extension='tif')

    @Dir
    def error(self):
        ...

    @Dir
    def intensity(self):
        ...

    @Dir
    def sidebyside(self):
        ...

    @Dir
    def soft(self):
        ...


class Padded(
    BaseGrid
):
    ...


padded = Padded()
BaseGrid.padded = padded
padded.__set_name__(BaseGrid, 'padded')
