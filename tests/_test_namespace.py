from __future__ import annotations
from tile2net.namespace import Namespace
import inspect
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

import shutil

import pytest
from tile2net.tileseg.raster import Raster

import ast
from types import SimpleNamespace


def find_namespace_attrs(code):
    """
    Parses Python code to find attributes accessed from a variable annotated as `Namespace`.
    Returns a set of attribute chains, e.g., 'model.eval'.
    """

    class NamespaceVisitor(ast.NodeVisitor):
        def __init__(self):
            self.namespace_attrs = set()

        def visit_Attribute(self, node):
            # Collect attribute access chains starting from Namespace variable
            chain = []
            while isinstance(node, ast.Attribute):
                chain.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name) and node.id == 'args':
                self.namespace_attrs.add('.'.join(reversed(chain)))
            self.generic_visit(node)

    tree = ast.parse(code)
    visitor = NamespaceVisitor()
    visitor.visit(tree)
    return visitor.namespace_attrs


def assert_namespace_attributes(code, Namespace):
    attrs = find_namespace_attrs(code)
    for attr in attrs:
        # For nested attributes like 'model.eval', split and check each part
        parts = attr.split('.')
        obj = Namespace
        for part in parts:
            assert hasattr(obj, part), f"Namespace does not have attribute '{part}'"
            obj = getattr(obj, part)  # Move to the next level


# Example usage with the provided code block
code_block = """
@commandline
def inference_(args: Namespace):
    # sys.stdin
    if args.dump_percent:
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        assert os.path.exists(args.result_dir), 'Result directory does not exist'
        logging.info(f'Inferencing. Segmentation results will be saved to {args.result_dir}')
    else:
        logging.info('Inferencing. Segmentation results will not be saved.')
"""


# # Mocking Namespace for demonstration. Replace with actual Namespace definition or another method to validate.
# Namespace = SimpleNamespace(result_dir="/path/to/results", dump_percent=True)
#
# # Perform assertion check
# assert_namespace_attributes(code_block, Namespace)

class NodeVisitor(ast.NodeVisitor):
    def __init__(self, tree):
        self.attrs = set()

    def visit_Attribute(self, node):
        # Collect attribute access chains starting from Namespace variable
        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if (
                isinstance(node, ast.Name)
                and node.id == 'args'
                and len(chain) > 0
        ):
            self.attrs.add('.'.join(reversed(chain)))
        self.generic_visit(node)



class FunctionVisitor:
    def __init__(self, function):
        self.function = function
        self.file = inspect.getsourcefile(function)

    @cached_property
    def code(self):
        return inspect.getsource(self.function)

    @cached_property
    def missing(self) -> list[str]:
        """
        Each attribute access to `args` that is not an
        attribute of the class Namespace

        e.g. args.result_dir is hasattr(Namespace, 'result_dir')
        e.g. args.model.eval is hasattr(args.model, 'eval')
        """

    @cached_property
    def functions(self) -> list[Callable]:
        """
        Each function that is called within self.function that
        contains the `args` variable;
        """


class FunctionsVisitor:
    def __init__(self, function):
        self.function = function

    @cached_property
    def missing(self) -> list[str]:
        stack = [self.function]
        visited = set()
        missing = set()
        while stack:
            function = stack.pop()
            ivistor = FunctionVisitor(function)
            visited.add(function)
            stack.extend(
                func
                for func in ivistor.functions
                if func not in visited
            )
            missing.update(ivistor.missing)


def missing_attributes(
        *functions: Callable,
        parameter: 'arg',
        namespace: Namespace
) -> list[str]:
    missing = []
    for function in functions:
        # assert that 'arg' is in the function signature
        visitor = FunctionVisitor(function)
        missing.extend(visitor.missing)




        # visitor = FunctionVisitor(function)
        # if parameter not in visitor.code:
        #     continue
        # missing = visitor.missing
        # if missing:
        #     return missing


if __name__ == '__main__':
    ...
