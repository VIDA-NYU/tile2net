from __future__ import annotations
from .intiles import InTiles

import os
import os.path

from .dir import Dir


class Output(
    Dir,
):
    ...

    # def iterator(self, dirname: str, *args, **kwargs) -> Iterator[pd.Series]:
    #     return super(Outputs, self).iterator(dirname)
    #     key = self._trace
    #     cache = self.tiles.attrs
    #     if key in cache:
    #         it = cache[key]
    #     else:
    #         files = self.files(dirname)
    #         if not self.tiles.cfg.force:
    #             loc = ~self.tiles.outdir.skip
    #             files = files.loc[loc]
    #         it = iter(files)
    #         cache[key] = it
    #     yield from it
    #     del cache[key]
    #


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
            'indexed',
            self.suffix.rsplit('.')[0] + '.png',
        )
        result = Indexed.from_format(format)
        return result

    @Colored
    def colored(self):
        format = os.path.join(
            self.dir,
            'colored',
            self.suffix.rsplit('.')[0] + '.png'
        )
        result = Colored.from_format(format)
        return result

    @InFile
    def infile(self):
        ...

        # format = os.path.join(
        #     self.dir,
        #     'infile',
        #     self.suffix.rsplit('.')[0] + '.png'
        # )
        # result = InFile.from_format(format)
        # return result

    @Overlay
    def overlay(self):
        format = os.path.join(
            self.dir,
            'overlay',
            self.suffix.rsplit('.')[0] + '.png',
        )
        result = Overlay.from_format(format)
        return result

    @Probability
    def prob(self):
        format = os.path.join(
            self.dir,
            'prob',
            self.suffix.rsplit('.')[0] + '.png',
        )
        result = Probability.from_format(format)
        return result

    @Error
    def error(self):
        format = os.path.join(
            self.dir,
            'error',
            self.suffix.rsplit('.')[0] + '.png',
        )
        result = Error.from_format(format)
        return result

    @Output
    def output(self):
        format = os.path.join(
            self.dir,
            'output',
            self.suffix.rsplit('.')[0] + '.png',
        )
        result = Output.from_format(format)
        return result
