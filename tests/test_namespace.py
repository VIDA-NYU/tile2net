from __future__ import annotations

import ast
import inspect
from typing import *

from tile2net.namespace import Namespace


class NodeVisitor(ast.NodeVisitor):
    def __init__(self, tree: ast.AST, parameter: str):
        self.args = set()
        self.parameter = parameter
        self.visit(tree)

    def visit_Attribute(self, node):
        # Collect attribute access chains starting from Namespace variable
        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if (
                isinstance(node, ast.Name)
                and node.id == self.parameter
                and len(chain)
        ):
            self.args.add('.'.join(reversed(chain)))
        self.generic_visit(node)


def missing_attributes(
        function: Callable,
        parameter: str = 'args',
        namespace: type = Namespace
) -> set:
    code = inspect.getsource(function)
    try:
        tree = ast.parse(code)
    except IndentationError:
        # Remove leading indentation
        code = '\n'.join(line[4:] for line in code.splitlines())
        tree = ast.parse(code)
    visitor = NodeVisitor(tree, parameter)
    missing = set()
    if not visitor.args:
        raise AttributeError(
            f'{parameter=} not found in {function.__name__} source code.'
        )
    for attr in visitor.args:
        obj = namespace
        parts = attr.split('.')
        for part in parts:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                missing.add(attr)
                break
    return missing


def should_fail(args: Namespace):
    args.not_real
    args.model.not_real


def test_namespace():
    from tile2net.tileseg.inference import inference
    from tile2net.tileseg.config import assert_and_infer_cfg
    from tile2net.tileseg.utils.misc import prep_experiment
    from tile2net.tileseg.datasets import setup_loaders
    from tile2net.tileseg.loss.utils import get_loss
    assert not missing_attributes(inference.inference)
    assert not missing_attributes(assert_and_infer_cfg)
    assert not missing_attributes(prep_experiment)
    assert not missing_attributes(setup_loaders)
    assert not missing_attributes(get_loss)
    assert not missing_attributes(inference.Inference.__init__)
    assert missing_attributes(should_fail)


if __name__ == '__main__':
    test_namespace()
