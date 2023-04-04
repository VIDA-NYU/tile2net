import os
from typing import Any, NamedTuple

import pytest
from toolz import pipe

from tile2net.namespace import AttrDict, Namespace

import logging


@pytest.fixture
def c() -> AttrDict:
    from tile2net.tileseg.config import cfg
    return cfg

@pytest.fixture
def args() -> Namespace:
    return Namespace(train_mode=True)

def test_found(c: AttrDict, args: Namespace):
    key: str
    left: Any | AttrDict
    not_found: list[str] = []
    mismatch: list[str] = []


    class Equivalent(NamedTuple):
        left: AttrDict
        left_name: str
        right: Namespace | AttrDict
        right_name: str


    E = Equivalent

    equivalents: list[Equivalent] = [E(c, '__C', args, 'args')]

    while equivalents:
        equivalent = equivalents.pop()
        for left_key, left in equivalent.left.items():
            right_key = left_key.lower()
            left_name = f'{equivalent.left_name}.{left_key}'
            right_name = f'{equivalent.right_name}.{right_key}'

            try:
                right = getattr(equivalent.right, right_key)
            except(AttributeError, KeyError):
                not_found.append(
                    f'{left_name} not found in \n\t{right_name}'
                )
                continue

            if isinstance(left, AttrDict):
                pipe(
                    E(left, left_name, right, right_name),
                    equivalents.append
                )
            elif left != right:
                if (
                        isinstance(left, str)
                        and os.sep in left
                ):
                    continue  # currently ignore paths
                mismatch.append(
                    f'{left_name}=({left}) != \n\t{right_name}=({right})'
                )
                # )

    for v in not_found:
        logging.error(v)
    for v in mismatch:
        logging.error(v)
    assert (
            not not_found
            and not mismatch
    )

def test_set(c: AttrDict, args: Namespace):
    c.immutable(False)
    args.immutable = False

    args.max_epoch = 1
    args.model.n_scales = 2
    args.model.ocr_extra.final_conv_kernel = 3
    args.model.ocr_extra.stage1.fuse_method = 4
    args.amsgrad = 5
    args.update_cfg()

    assert c.MAX_EPOCH == 1
    assert c.MODEL.N_SCALES == 2
    assert c.MODEL.OCR_EXTRA.FINAL_CONV_KERNEL == 3
    assert c.MODEL.OCR_EXTRA.STAGE1.FUSE_METHOD == 4
    assert c.AMSGRAD == 5
    pass

if __name__ == '__main__':
    from tile2net.tileseg.config import cfg
    args = Namespace()
    test_found(cfg, args)
    test_set(cfg, args)
    # pytest.main('-vv namespace_test.py::test_set'.split())
    # pytest.main('-vv namespace_test.py::test_found'.split())
