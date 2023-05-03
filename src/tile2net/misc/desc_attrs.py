from __future__ import annotations

import os
from typing import Callable, Type
from weakref import WeakKeyDictionary

from tile2net.misc.attrs import attr


class desc_attr(attr):
    instance: Descriptor
    owner: Type[Descriptor]

    def __hash__(self):
        return hash(self.instance)

    def __eq__(self, other):
        return self.instance == other.instance

    def __set_name__(self, owner, name):
        self.cache = WeakKeyDictionary()
        super().__set_name__(owner, name)

    def __get__(self, instance: Descriptor, owner):
        return self.cache[self]

    def __set__(self, instance: Descriptor, value):
        self.cache[self] = value

    def __delete__(self, instance: Descriptor):
        del self.cache[self]

    def __bool__(self):
        return self in self.cache

    def __fspath__(self):
        return os.path.join(
            self.instance._artifacts.__fspath__(),
            self.name + '.pkl'
        )

class desc_subframe(desc_attr):
    __set_name__ = desc_attr.__set_name__

if __name__ == '__main__':
    from tile2net.artifacts.descriptor import Descriptor
    from pandas import DataFrame
    import numpy as np
    from numpy import ndarray


    class TestDesc(Descriptor):
        @desc_attr
        @property
        def attr(self):
            print('attr')
            return False

        @desc_subframe
        @property
        def subframe(self):
            print('subframe')
            return self._artifacts.copy()

    class TestFrame(DataFrame):
        desc = TestDesc()

        @classmethod
        def from_test(cls):
            return cls({'a': np.arange(10)})

        @property
        def _constructor(self) -> Callable[..., DataFrame]:
            return TestFrame

    test = TestFrame.from_test()
    test.desc.attr
    test.desc.subframe
    sub = test.loc[5:]
    sub.desc.attr
    sub.desc.subframe
    pass
