from __future__ import annotations
from ..grid import filled

from .ingrid import InGrid

if False:
    from ..seggrid.seggrid import SegGrid


class Filled(
    filled.Filled,
    InGrid,
):

    def _get(
            self: Filled,
            instance: InGrid,
            owner,
    ) -> Filled:
        if instance is None:
            return self
        # instance = instance.ingrid
        # self.instance = instance
        cache = instance.frame.__dict__
        key = self.__name__

        if key in cache:
            result = cache[key]
        else:
            seggrid = instance.seggrid
            result = (
                instance
                .to_scale(seggrid.scale)
                .to_padding(instance.cfg.segmentation.pad)
                .to_scale(instance.scale)
                .pipe(self.__class__.from_wrapper)
            )
            assert isinstance(result, self.__class__)

            result.__dict__.update(instance.__dict__)
            result.instance = instance
            instance.frame.__dict__[self.__name__] = result
        return result

    locals().update(
        __get__=_get
    )

    @property
    def seggrid(self) -> SegGrid:
        return self.instance.seggrid

    @property
    def filled(self):
        return self.instance.filled

    @property
    def ingrid(self) -> InGrid:
        return self.instance
