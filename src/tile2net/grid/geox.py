
from __future__ import annotations

import re
from typing import TypeVar

import pyproj

T = TypeVar('T')


class GeoX:
    """
    Allows the user to more conveniently perform a spatial query;
    GDF.cx[...] has the requirements that the coordinates match
    the GDF crs, and that the coordinates are in the order (x, y).
    This method allows the user to use (y, x) coordinates, and
    automatically transforms from 4326 to the GDF crs.
    """
    _regex = re.compile(r'(-?\d+\.?\d*),(-?\d+\.?\d*)')

    def __init__(self, *args):
        ...

    # todo: prevent memory leak; just return a copy with self.__self__ = instance
    def __get__(self, instance: NDFrame, owner):
        self.instance = instance
        self.owner = owner
        return self

    def __getitem__(self, item):
        """
        artifacts.geox['y,x':'y,x']
        artifacts.geox[(y,x):(y,x)]
        artifacts.geox[y,x:y,x]
        artifacts.geox[y,x,y,x]
        :return: Source
        """
        instance = self.instance

        if isinstance(item, slice):
            start = item.start
            stop = item.stop
            if isinstance(start, str):
                # todo
                raise NotImplementedError('my regex is apparently wrong')
                match = self._regex._match(start)
                s = float(match.subnode(1))
                w = float(match.subnode(2))
            else:
                s, w = start
            if isinstance(stop, str):
                match = self._regex._match(stop)
                n = float(match.subnode(1))
                e = float(match.subnode(2))
            else:
                n, e = stop

        elif (
                isinstance(item, tuple)
                and len(item) == 3
                and isinstance(item[1], slice)
        ):
            s = item[0]
            w = item[1].start
            n = item[1].stop
            e = item[2]

        elif (
                isinstance(item, tuple)
                and len(item) == 4
        ):
            s, w, n, e = item

        else:
            raise ValueError('invalid refx slice: %s' % item)

        w, e = min(w, e), max(w, e)
        s, n = min(s, n), max(s, n)
        trans = pyproj.Transformer.from_crs(4326, instance.crs, always_xy=True).transform
        w, s = trans(w, s)
        e, n = trans(e, n)
        result = instance.cx[w:e, s:n]
        return result


