from __future__ import annotations
from .intiles import  InTiles

import os
import os.path

from .dir import Dir


class Probability(
    Dir,
):
    extension = 'npy'


class Prediction(
    Dir
):
    extension = 'png'


class Error(
    Dir
):
    extension = 'npy'


class InFile(
    Dir
):
    ...


class Indexed(
    Dir
):
    ...


class Colored(
    Dir
):
    ...


class Overlay(
    Dir
):
    ...


class SegTiles(
    Dir,
):

    @Indexed
    def indexed(self):
        format = os.path.join(
            self.dir,
            'prediction',
            self.suffix
        ).replace(self.extension, 'png')
        result = Indexed.from_format(format)
        return result

    @Colored
    def colored(self):
        format = os.path.join(
            self.dir,
            'mask',
            self.suffix
        ).replace(self.extension, 'png')
        result = Colored.from_format(format)
        return result

    @InFile
    def infile(self):
        format = os.path.join(
            self.dir,
            'infile',
            self.suffix
        ).replace(self.extension, 'png')
        result = InFile.from_format(format)
        return result

    @Overlay
    def overlay(self):
        format = os.path.join(
            self.dir,
            'overlay',
            self.suffix
        ).replace(self.extension, 'png')
        result = Overlay.from_format(format)
        return result

    @Probability
    def prob(self):
        format = os.path.join(
            self.dir,
            'prob',
            self.suffix,
        ).replace(self.extension, 'png')
        result = Probability.from_format(format)
        return result

    @Error
    def error(self):
        format = os.path.join(
            self.dir,
            'error',
            self.suffix,
        ).replace(self.extension, 'png')
        result = Error.from_format(format)
        return result
