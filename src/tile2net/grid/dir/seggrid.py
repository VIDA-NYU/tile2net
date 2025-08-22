from __future__ import annotations

from .dir import Dir


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


class Grayscale(
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


class SegGrid(
    Dir,
):
    padded: Padded

    @Grayscale
    def grayscale(self):
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
        ...

    @Error
    def error(self):
        ...

    @Output
    def output(self):
        ...

    @property
    def pickle(self) -> str:
        result = f'{self.dir}/seggrid.pickle'
        return result


class Padded(
    SegGrid
):
    ...

padded = Padded()
SegGrid.padded = padded
padded.__set_name__(SegGrid, 'padded')
