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
