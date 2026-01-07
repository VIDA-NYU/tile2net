from __future__ import annotations

from .dir import Dir
import os


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


class static(
    Dir
):
    ...


class Pred(
    Dir
):
    extension = 'npy'


class colorized(
    Dir
):
    ...


class Intensity(
    Dir
):
    ...


class Overlay(
    Dir
):
    ...


class Postprocess(
    Dir
):
    grid: Grid

    @Pred
    def pred(self):
        ...

    @colorized
    def colorized(self):
        ...

    @Overlay
    def overlay(self):
        ...

    @Probability
    def prob(self):
        ...

    @Error
    def error(self):
        ...

    @Intensity
    def intensity(self):
        ...


class Grid(
    Dir,
):
    padded: Padded

    @Postprocess
    def postprocess(self):
        ...

    @Pred
    def pred(self):
        ...

    @colorized
    def colorized(self):
        ...

    @static
    def static(self):
        ...

    @Overlay
    def overlay(self):
        ...

    @Probability
    def prob(self):
        format = os.path.join(
            self.dir,
            'prob',
            self.suffix + '.tif',
        )
        format = format
        result = Probability.from_format(format)
        return result

    @Error
    def error(self):
        ...

    @Intensity
    def intensity(self):
        ...


class Padded(
    Grid
):
    ...


padded = Padded()
Grid.padded = padded
padded.__set_name__(Grid, 'padded')
