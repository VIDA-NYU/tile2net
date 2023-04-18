from weakref import WeakKeyDictionary

import toolz
import inspect
import json


def southwest_northeast(bbox: list[float]):
    return [
        min(bbox[0], bbox[2]),
        min(bbox[1], bbox[3]),
        max(bbox[0], bbox[2]),
        max(bbox[1], bbox[3]),
    ]

def unpack_relevant(cls, info) -> object:
    if not isinstance(info, dict):
        with open(info) as f:
            kwargs = json.load(f)
    else:
        kwargs = info
    relevant = toolz.keyfilter(
        inspect.signature(cls.__init__).parameters.__contains__,
        kwargs
    )
    res = cls(**relevant)
    return res

class cached_descriptor(property):

    def __init__(self, fget):
        super().__init__(fget)
        self.cache = WeakKeyDictionary()

    # noinspection PyMethodOverriding
    def __get__(self, instance, owner):
        if instance is None:
            return self
        if instance not in self.cache:
            self.cache[instance] = self.fget(instance)
        return self.cache[instance]

    def __set__(self, instance, value):
        self.cache[instance] = value

    def __delete__(self, instance):
        del self.cache[instance]
