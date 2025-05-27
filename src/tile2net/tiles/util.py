import math

import numpy as np
from numpy import ndarray
import numba as nb
from functools import singledispatch

if False:
    import folium


@singledispatch
def num2deg(
        x: float,
        y: float,
        zoom: int,
) -> tuple[float, float]:
    n = 2.0 ** zoom
    lon = (x / n * 360.0) - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


@num2deg.register
def _(
        x: ndarray,
        y: ndarray,
        zoom: int,
) -> tuple[ndarray, ndarray]:
    n = 2.0 ** zoom
    lon = (x / n * 360.0) - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
    return lon, lat


@singledispatch
def deg2num(
        lon: float,
        lat: float,
        zoom: int,
) -> tuple[int, int]:
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(
        math.tan(lat * math.pi / 180.0) + 1.0 / math.cos(lat * math.pi / 180.0)) / math.pi) / 2.0 * n)
    return x, y


@deg2num.register
def _(
        lon: ndarray,
        lat: ndarray,
        zoom: int,
) -> tuple[
    ndarray[int],
    ndarray[int],
]:
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.log(
        np.tan(lat * np.pi / 180.0) + 1.0 / np.cos(lat * np.pi / 180.0)) / np.pi) / 2.0 * n
    return x.astype(int), y.astype(int)


def import_folium() -> 'folium':
    # When debugging, importing folium seems to cause AttributeError: module 'posixpath' has no attribute 'sep'
    import posixpath
    sep = posixpath.sep
    import folium
    posixpath.sep = sep
    return folium
