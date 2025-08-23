from __future__ import annotations

import os
import os.path

from .dir import Dir
from .sourcedir import SourceDir

if False:
    import tile2net.grid.ingrid


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


class SideBySide(
    Dir
):
    ...


# class SegResults(
#     Dir,
# ):
#
#     @Probability
#     def prob(self):
#         format = os.path.join(
#             self.dir,
#             'prob',
#             self.suffix,
#         ).replace(self.extension, 'png')
#         result = Probability.from_format(format)
#         return result
#
#     @Error
#     def error(self):
#         format = os.path.join(
#             self.dir,
#             'error',
#             self.suffix,
#         ).replace(self.extension, 'png')
#         result = Error.from_format(format)
#         return result
#
#     @SideBySide
#     def sidebyside(self):
#         format = os.path.join(
#             self.dir,
#             'sidebyside',
#             self.suffix,
#         ).replace(self.extension, 'png')
#         result = SideBySide.from_format(format)
#         return result
#
#     @property
#     def topn_failures(self) -> str:
#         return os.path.join(self.dir, 'topn_failures.html')
#

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
    grid: tile2net.grid.ingrid.InGrid

    # @Outputs
    # def outputs(self):
    #     format = os.path.join(
    #         self.dir,
    #         'outputs',
    #         self.suffix,
    #     )
    #     result = Outputs.from_format(format)
    #     return result

    # @Submit
    # def submit(self):
    #     format = os.path.join(
    #         self.dir,
    #         'submit',
    #         self.suffix,
    #     )
    #     result = Submit.from_format(format)
    #     return result
    #
    #
    # @BestImages
    # def best_images(self):
    #     format = os.path.join(
    #         self.dir,
    #         'best_images',
    #         self.suffix,
    #     )
    #     result = BestImages.from_format(format)
    #     return result

    @SourceDir
    def sourcedir(self):
        grid = self.grid
        name = None
        try:
            name = grid.source.name
        except ValueError:
            ...
        if name is None:
            name = grid.cfg.indir.name
        if name is None:
            name = grid.indir.dir.rsplit(os.sep, 1)[-1]
        format = os.path.join(
            self.dir,
            name,
            self.suffix
        )
        result = SourceDir.from_format(format)
        return result

    @property
    def lines(self):
        return self.sourcedir.lines

    @property
    def polygons(self):
        return self.sourcedir.polygons

    @property
    def vecgrid(self):
        return self.sourcedir.namedir.vecgrid

    @property
    def seggrid(self):
        return self.sourcedir.namedir.seggrid

    @property
    def namedir(self):
        return self.sourcedir.namedir

    @property
    def ingrid(self):
        return self.sourcedir.ingrid
