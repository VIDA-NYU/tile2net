from __future__ import annotations

from tile2net.grid.dir.dir import Dir


class Outputs(
    Dir
):
    basegrid: BaseGrid

    @Dir
    def pred(self):
        return Dir.from_parent(self, 'prob', extension='tif')

    @Dir
    def colorized(self):
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
    def static(self):
        ...

    @Dir
    def sidebyside(self):
        ...

    @Dir
    def soft(self):
        ...


class BaseGrid(
    Outputs
):
    padded: Padded

    @Outputs
    def dense_crf(self):
        ...

    @Outputs
    def guided_filter(self):
        ...

    @Outputs
    def slic(self):
        ...

    @Outputs
    def walker(self):
        ...


class Padded(
    BaseGrid
):
    ...


padded = Padded()
BaseGrid.padded = padded
padded.__set_name__(BaseGrid, 'padded')
