import ast
import inspect
import json
import os
import textwrap
from functools import singledispatch
from typing import *
from weakref import WeakKeyDictionary

import geopy
import math
import numpy as np
import toolz
from geopy.geocoders import Nominatim
from numpy import ndarray
from toolz import curried, pipe

from tile2net.tiles.cfg.logger import logger

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
    ndarray,
    ndarray,
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


def round_loc(location: list[float], decimals=10) -> list[float]:
    return list(np.around(np.array(location), decimals=decimals))


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


def geocode(location) -> list[float]:
    # from address, get bbox
    if isinstance(location, str):
        try:
            location: list[float] = pipe(
                location.split(','),
                curried.map(float),
                list
            )
        except (ValueError, AttributeError):  # fails if address or list
            msg = f'Geocoding the following, please wait:\n \t{location}'
            logger.info(msg)
            nom: geopy.Location = Nominatim(user_agent='tile2net').geocode(location, timeout=None)
            if nom is None:
                raise ValueError(f"Could not geocode '{location}'")
            logger.info(f"Geocoded to\n\t{nom.raw['display_name']}")
            location = pipe(
                nom.raw['boundingbox'],
                # convert lon, lon, lat, lat
                # to lat, lon, lat, lon
                curried.get([0, 2, 1, 3]),
            )
    location = pipe(
        location,
        curried.map(float),
        list,
        round_loc,
        southwest_northeast,
        tuple
    )
    return location


def reverse_geocode(location: list[float]) -> str:
    # from bbox, get address of centroid
    msg = f'Reverse geocoding the following, please wait:\n \t{location}'
    logger.info(msg)
    # nom: geopy.Location = Nominatim(user_agent='tile2net').reverse(location, timeout=None)
    y = (location[0] + location[2]) / 2
    x = (location[1] + location[3]) / 2
    centroid = (y, x)
    nom: geopy.Location = Nominatim(
        user_agent='tile2net',
    ).reverse(centroid, timeout=None)
    result = nom.raw['display_name']
    return result

    if nom is None:
        raise ValueError(f"Could not geocode '{location}'")
    logger.info(f"Geocoded to\n\t{nom.raw['display_name']}")
    return nom.raw['display_name']


def name_from_location(location: str | list[float, str]):
    if isinstance(location, str):
        try:
            # location is bbox
            location = pipe(
                location.split(','),
                curried.map(float),
                list,
            )
        except (ValueError, AttributeError):  # fails if already address
            # location is address
            ...
    if isinstance(location, list):
        # location is bbox
        centroid = (
            (location[0] + location[2]) / 2,
            (location[1] + location[3]) / 2,
        )
        msg = f'Reverse geocoding the following, please wait:\n \t{centroid}'
        logger.info(msg)
        nom: geopy.Location = Nominatim(user_agent='tile2net').reverse(centroid, timeout=None)
        logger.info(f"Geocoded  to\n\t'{nom.raw['display_name']}'")
        location = nom.raw['display_name']

    if isinstance(location, str):
        # location is address
        name = pipe(
            location.split(',')[0]
            .replace(' ', '_')
            .casefold(),
            os.path.normcase
        )
        return name
    raise TypeError(f"location must be str or list, not {type(location)}")


def has_uncommented_return(func):
    source_code = inspect.getsource(func)
    source_code = textwrap.dedent(source_code)
    tree = ast.parse(source_code)

    class ReturnVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Return(self, node):
            if (
                    not isinstance(node.value, ast.Constant)
                    or node.value.value is not None
            ):
                self.found = True

    visitor = ReturnVisitor()
    visitor.visit(tree)
    return visitor.found


class LazyModuleLoader:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def _load_module(self):
        if self.module is None:
            self.module = __import__(self.module_name, fromlist=[''])

    def __getattr__(self, item):
        self._load_module()
        return getattr(self.module, item)

    def __setattr__(self, key, value):
        if key in ('module_name', 'module'):
            super().__setattr__(key, value)
        else:
            self._load_module()
            setattr(self.module, key, value)


def noreturn(func) -> bool:
    tree = ast.parse(inspect.getsource(func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            return True
    return False


class ReturnVisitor(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Return(self, node):
        self.found = True
        self.generic_visit(node)


class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_code = False

    def visit_Assign(self, node):
        self.has_code = True
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.has_code = True
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.has_code = True
        self.generic_visit(node)

    def visit_Return(self, node):
        self.has_code = True
        self.generic_visit(node)


def contains_functioning_code(code):
    if inspect.isfunction(code):
        code = inspect.getsource(code)
    dedented_code = textwrap.dedent(code)
    tree = ast.parse(dedented_code)
    visitor = CodeVisitor()
    visitor.visit(tree)
    return visitor.has_code


def returns_or_assigns(code) -> bool:
    if not code:
        return False
    return (
            returns(code)
            or contains_functioning_code(code)
    )


def returns(code):
    if inspect.isfunction(code):
        code = inspect.getsource(code)
    dedented_code = textwrap.dedent(code)
    tree = ast.parse(dedented_code)
    visitor = ReturnVisitor()
    visitor.visit(tree)
    return visitor.found


def has_executable_code(func):
    tree = ast.parse(inspect.getsource(func))
    for node in ast.walk(tree):
        clses = (
            ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.While,
            ast.If, ast.With, ast.Call, ast.Expr, ast.AsyncFor,
            ast.AsyncWith, ast.Try, ast.ExceptHandler, ast.FunctionDef, ast.ClassDef,
        )
        if isinstance(node, clses):
            return True
    return False


T = TypeVar('T')


def _look_at(func: T) -> T:
    return func


def look_at(file: object):
    return _look_at


if __name__ == '__main__':
    print(name_from_location('New York, NY, USA'))
    print(name_from_location([1.22456789, 2.3456789, 3.456789, 4.56789]))
