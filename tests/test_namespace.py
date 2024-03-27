from __future__ import annotations
import itertools

import itertools
from collections import defaultdict
import re
import numpy as np
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd
from functools import *
from typing import *
from types import *
from shapely import *
import magicpandas as magic

import ast
import inspect
import os.path
from typing import *

from tile2net.namespace import Namespace

# class NodeVisitor(ast.NodeVisitor):
#     def __init__(self, tree: ast.AST, parameter: str):
#         self.args = set()
#         self.parameter = parameter
#         self.visit(tree)
#
#     def visit_Attribute(self, node):
#         # Collect attribute access chains starting from Namespace variable
#         chain = []
#         while isinstance(node, ast.Attribute):
#             chain.append(node.attr)
#             node = node.value
#         if (
#                 isinstance(node, ast.Name)
#                 and node.id == self.parameter
#                 and len(chain)
#         ):
#             self.args.add('.'.join(reversed(chain)))
#         self.generic_visit(node)
#
#
# def missing_attributes(
#         function: Callable,
#         parameter: str = 'args',
#         namespace: type = Namespace
# ) -> set:
#     code = inspect.getsource(function)
#     try:
#         tree = ast.parse(code)
#     except IndentationError:
#         # Remove leading indentation
#         code = '\n'.join(line[4:] for line in code.splitlines())
#         tree = ast.parse(code)
#     visitor = NodeVisitor(tree, parameter)
#     missing = set()
#     if not visitor.args:
#         raise AttributeError(
#             f'{parameter=} not found in {function.__name__} source code.'
#         )
#     for attr in visitor.args:
#         obj = namespace
#         parts = attr.split('.')
#         for part in parts:
#             try:
#                 obj = getattr(obj, part)
#             except AttributeError:
#                 missing.add(attr)
#                 break
#     return missing


"""
(args.hello)    yields args.hello
something.args.hello    yields nothing
print(hello, args.hello)    yields args.hello
[args.hello.world] yields args.hello.world
args.hello.world()      yields args.hello.world
"""


class AttributeAccesses:
    regex = r'\bargs[a-zA-Z.]+[a-zA-Z]\b'

    # regex = r""
    # regex = r"(?<!#.*)\bargs[a-zA-Z.]+[a-zA-Z]\b"

    @classmethod
    def test_regex(cls):
        regex = cls.regex
        assert re.search(regex, 'args.hello')
        assert re.search(regex, 'args.hello.world()')
        assert re.search(regex, 'args.hello')
        assert re.search(regex, 'args.hello.world').group(0)
        assert re.search(regex, 'args.hello.world()').group(0)
        assert re.search(regex, '[args.hello.world]').group(0)
        assert not re.search(regex, 'kwargs.hello.world')
        assert not re.search(regex, 'bargs.hello')

    def __init__(
            self,
            top: str,
            name: str,
            namespace: type[Namespace] = Namespace,
            exclude: set[str] = frozenset()
    ):
        self.top = top
        self.name = name
        self.namespace = namespace
        self.exclude = exclude

    @cached_property
    def accesses(self) -> dict:
        top = self.top
        regex = self.regex
        accesses = defaultdict(lambda: defaultdict(set))
        for root, dirs, files in os.walk(top):
            for file in files:
                if not file.endswith('.py'):
                    continue
                path = os.path.join(root, file)
                if any(exclude in path for exclude in self.exclude):
                    continue
                with open(path) as f:
                    for n, line in enumerate(f, 1):
                        if line.strip().startswith('#'):
                            continue
                        matches = re.findall(regex, line)
                        if len(matches):
                            accesses[path][n].update(matches)

        return accesses

    @cached_property
    def misses(self) -> dict:
        result = {}

        exceptions = set(itertools.chain(
            dir(int),
            dir(float),
            dir(str),
            dir(bool),

        ))

        # for path, (line, matches) in self.accesses.items():
        result = defaultdict(lambda: defaultdict(set))
        for path, line_matches in self.accesses.items():
            for line, matches in line_matches.items():
                namespace = self.namespace
                for match in matches:
                    obj = namespace
                    parts = match.split('.')[1:]
                    for part in parts:
                        try:
                            obj = getattr(obj, part)
                        except AttributeError:
                            if part in exceptions:
                                continue
                            result[path][line].add(match)
                            break
        return result

def test_namespaces():
    top = os.path.join(
        __file__,
        '..',
        '..',
        'src',
        'tile2net',
    )
    top = os.path.abspath(top)
    attrs = AttributeAccesses(
        top=top,
        name='args',
        exclude={'tile2net/src/tile2net/tileseg/tests/', }
    )
    assert not attrs.misses


if __name__ == '__main__':
    test_namespaces()

# def should_fail(args: Namespace):
#     args.not_real
#     args.model.not_real
#
#
# def test_namespace():
#     assert not missing_attributes('inference.py')
#     from tile2net.tileseg.inference import inference
#     from tile2net.tileseg.config import assert_and_infer_cfg
#     from tile2net.tileseg.utils.misc import prep_experiment
#     from tile2net.tileseg.datasets import setup_loaders
#     from tile2net.tileseg.loss.utils import get_loss
#     assert not missing_attributes(inference.inference)
#     assert not missing_attributes(assert_and_infer_cfg)
#     assert not missing_attributes(prep_experiment)
#     assert not missing_attributes(setup_loaders)
#     assert not missing_attributes(get_loss)
#     assert not missing_attributes(inference.Inference.__init__)
#     assert missing_attributes(should_fail)
#
#
