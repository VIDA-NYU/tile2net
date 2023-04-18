from __future__ import annotations


if False:
    from tile2net.raster.raster import Raster

import functools


class validate:
    def __set_name__(self, owner, name):
        self.name = name

    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, instance: Raster, owner: type[Raster]):
        self.raster = instance
        self.Raster = owner
        return self

    def __call__(self, *args, **kwargs):
        raster = self.raster
        # if self.name == 'generate':
        #     print('Generating')
        #
        # if self.name == 'inference':
            # print('Inference')

        result =  self.func(raster, *args, **kwargs)

        # print('Done')

        return result

if __name__ == '__main__':
    class Raster:
        @validate
        def generate(self):
            ...

        @validate
        def inference(self):
            ...

    r = Raster()
    r.generate()
    r.inference()
