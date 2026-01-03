from __future__ import annotations

from .dir import Dir
import os


class Output(
    Dir,
):
    ...

    # def iterator(self, dirname: str, *args, **kwargs) -> Iterator[pd.Series]:
    #     return super(Outputs, self).iterator(dirname)
    #     key = self._trace
    #     cache = self.grid.__dict__
    #     if key in cache:
    #         it = cache[key]
    #     else:
    #         files = self.files(dirname)
    #         if not self.grid.cfg.force:
    #             loc = ~self.grid.outdir.skip
    #             files = files.loc[loc]
    #         it = iter(files)
    #         cache[key] = it
    #     yield from it
    #     del cache[key]
    #


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


class InFile(
    Dir
):
    ...


class Pred(
    Dir
):
    extension = 'npy'


class Colored(
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
    grid: SegGrid

    @Pred
    def pred(self):
        ...

    @Colored
    def colored(self):
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

    @Output
    def output(self):
        ...

    @Intensity
    def intensity(self):
        ...


class SegGrid(
    Dir,
):
    padded: Padded

    @Postprocess
    def postprocess(self):
        ...

    @Pred
    def pred(self):
        ...

    @Colored
    def colored(self):
        ...

    @InFile
    def infile(self):
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

    @Output
    def output(self):
        ...

    @Intensity
    def intensity(self):
        ...

    @property
    def pickle(self) -> str:
        result = f'{self.dir}/seggrid.pickle'
        return result

    @property
    def summary(self) -> str:
        result = f'{self.dir}/summary.txt'
        return result


class Padded(
    SegGrid
):
    ...


padded = Padded()
SegGrid.padded = padded
padded.__set_name__(SegGrid, 'padded')
