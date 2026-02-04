from __future__ import annotations

from tile2net.core.dir.dir import Dir


class Outputs(
    Dir
):
    grid: Grid

    @Dir
    def pred(self):
        return Dir.from_parent(self, 'pred', extension='tif')

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
    def unclipped_prob(self):
        return Dir.from_parent(self, 'unclipped_prob', extension='tif')

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

    @Dir
    def mask(self):
        ...


class PostProcessedOutputs(
    Outputs
):
    @Dir
    def colorized_sidebyside(self):
        ...

    @Dir
    def comparison(self):
        ...


class Grid(
    Outputs
):
    padded: Padded

    @PostProcessedOutputs
    def dense_crf(self):
        ...

    @PostProcessedOutputs
    def test(self):
        ...

    @PostProcessedOutputs
    def slic(self):
        ...

    @PostProcessedOutputs
    def walker(self):
        ...

    @PostProcessedOutputs
    def hysteresis(self):
        ...

    @PostProcessedOutputs
    def gac(self):
        ...

    @PostProcessedOutputs
    def gmb(self):
        ...


class Padded(
    Grid
):
    ...


padded = Padded()
Grid.padded = padded
padded.__set_name__(Grid, 'padded')
