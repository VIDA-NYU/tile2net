"""

# Code adapted from:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py

Source License
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
import functools
import json
import os
import re
import select

import sys
from functools import cached_property
from typing import Any, Iterator, NamedTuple, Optional, Type

import argh
import argh.constants
import itertools
import torch
from runx.logx import logx
from toolz import pipe

from tile2net.tileseg.config import cfg
from tile2net.tileseg.utils.attr_dict import AttrDict
import toolz
# import logging
from tile2net.logger import logger
from tile2net.raster.project import Project


def torch_version_float():
    version_str = torch.__version__
    version_re = re.search(r'^([0-9]+\.[0-9]+)', version_str)
    if version_re:
        version = float(version_re.group(1))
        logx.msg(f'Torch version: {version}, {version_str}')
    else:
        version = 1.0
        logx.msg(f'Can\'t parse torch version ({version}), assuming {version}')
    return version


class Immutability:
    _mutable: set[str] = 'immutable _immutable'.split()
    _immutable = False

    @cached_property
    def _children(self):
        return tuple([
            v
            for v in self.__dict__.values()
            if isinstance(v, Immutability)
        ])

    @property
    def immutable(self):
        return self._immutable

    @immutable.setter
    def immutable(self, value):
        self._immutable = value
        for child in self._children:
            child.immutable = value

    def __init_subclass__(cls, **kwargs):
        mutable: set = {
            m
            for base in cls.__bases__
            if issubclass(base, Immutability)
            for m in base._mutable
        }
        if '_mutable' in cls.__dict__:
            mutable.update(cls.__dict__['_mutable'])
            # mutable.add(cls.__dict__['_mutable'])
        setattr(cls, '_mutable', mutable)
        super().__init_subclass__(**kwargs)


    def __setattr__(self, key, value):
        if (
                self.immutable
                and key not in self._mutable
        ):
            raise AttributeError(
                f'Attempted to set "{key}" to "{value}", but AttrDict is immutable'
            )
        super().__setattr__(key, value)

# class AttrDesc(Immutability, AttrDict):
#     _mutable = '_instance _owner'.split()
#
#     def __get__(self, instance, owner):
#         self._instance = instance
#         self._owner = owner
#         return self
#
#     def __set_name__(self, owner, name):
#         self._name = name
#
#     def __repr__(self):
#         return self._name

class AttrDesc(Immutability):
    _mutable = '_instance _owner'.split()

    def __get__(self, instance, owner):
        self._instance = instance
        self._owner = owner
        return self

    def __set_name__(self, owner, name):
        self._name = name

    def __repr__(self):
        return self._name


class Options(AttrDesc):
    test_mode = None
    init_decoder = None
    torch_version = None

    def __get__(self, instance, owner) -> 'Options':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class Train(AttrDesc):
    fp16 = None
    random_brightness_shift_value = 10

    def __get__(self, instance, owner) -> 'Train':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class Dataset(AttrDesc):
    _mutable = 'num_classes ignore_label'.split()

    def __get__(self, instance, owner) -> 'Dataset':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

    cityscapes_dir = None
    cityscapes_customcoarse = None
    centroid_root = None
    cityscapes_aug_dir = None
    satellite_dir = None

    camvid_dir = None
    cityscapes_splits = None
    class_uniform_pct = None
    class_uniform_tile = None
    coarse_boost_classes = None
    colorize_mask_fn = None
    # crop_size = None
    crop_size = None
    custom_coarse_dropout_classes: str = None
    cv = None
    dump_images = None
    ignore_label = None
    kitti_aug_dir = None
    lanczos_scales = None
    mapillary_crop_val = None
    mask_out_cityscapes = None
    mean = None
    name = None
    num_classes = None
    std = None
    # todo centroid root, satellite_dir
    translate_aug_fix = None

class Stage1(AttrDesc):
    block = None
    fuse_method = None
    num_blocks = None
    num_channels = None
    num_modules = None
    num_ranches = None

    def __get__(self, instance, owner) -> 'Stage1':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class Stage2(AttrDesc):

    block = None
    fuse_method = None
    num_blocks = None
    num_branches = None
    num_channels = None
    num_modules = None

    def __get__(self, instance, owner) -> 'Stage2':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class Stage3(AttrDesc):
    num_modules = None
    num_branches = None
    block = None
    num_blocks = None
    num_channels = None
    fuse_method = None


    def __get__(self, instance, owner) -> 'Stage3':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class Stage4(AttrDesc):

    block = None
    fuse_method = None
    num_blocks = None
    num_branches = None
    num_channels = None
    num_modules = None

    def __get__(self, instance, owner) -> 'Stage4':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class Ocr(AttrDesc):
    key_channels = None
    mid_channels = None

    def __get__(self, instance, owner) -> 'Ocr':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class OcrExtra(AttrDesc):
    stage1 = Stage1()
    stage2 = Stage2()
    stage3 = Stage3()
    stage4 = Stage4()
    final_conv_kernel = None

    def __get__(self, instance, owner) -> 'OcrExtra':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class Model(AttrDesc):
    ocr_extra = OcrExtra()
    ocr = Ocr()

    scale_min: float = None
    align_corners = None
    alt_two_scale = None
    aspp_bot_ch = None
    attnscale_bn_head = None
    bnfunc = None
    bn = None
    extra_scales = None
    grad_ckpt = None
    lr_scheduler: str = None
    mscale_cat_scale_flt = None
    mscale_dropout = None
    mscale = None
    mscale_init = None
    mscale_inner_3x3 = None
    mscale_lo_scale = None
    mscale_oldarch = None
    n_scales = None
    ocr_aspp = None
    optimizer: str = None
    rmi_loss = None
    segattn_bot_ch = None
    three_scale = None
    bs_trn: int = None
    img_wt_loss: bool = None
    bs_val: int = None
    color_aug: float = None
    gblur: bool = None
    bblur: bool = None
    full_crop_modeling = None
    eval: Optional[str] = None
    rand_augment: Optional[str] = None
    scale_max: float = None
    pre_size: Optional[int] = None

    weights_path: str = None
    wrn38_checkpoint: str = None
    wrn41_checkpoint: str = None
    x71_checkpoint: str = None
    hrnet_checkpoint: str = None
    snapshot: str = None

    def __get__(self, instance, owner) -> 'Model':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

class Loss(AttrDesc):
    ocr_alpha = None
    ocr_aux_rmi = None
    supervised_mscale_wt = None

    def __get__(self, instance, owner) -> 'Loss':
        # noinspection PyTypeChecker
        return super().__get__(instance, owner)

# class Singleton:
#     def __new__(cls, *args, **kwargs):
#         try:
#             instance = getattr(cls, '_instance')
#         except AttributeError:
#             instance = super().__new__(cls, *args, **kwargs)
#             setattr(cls, '_instance', instance)
#         return instance
#
# class Namespace(Singleton, Immutability, argh.ArghNamespace):
class Namespace(
    # Singleton,
    Immutability,
    argh.ArghNamespace,
):
    options = Options()
    train = Train()
    dataset = Dataset()
    loss = Loss()
    model = Model()

    eval_folder: str = None

    _assets_path: str = None

    @property
    def assets_path(self):
        return self._assets_path

    @assets_path.setter
    def assets_path(self, value):
        assets = self._assets_path = value

        self.dataset.centroid_root = os.path.join(assets, 'uniform_centroids')
        self.model.weights_path = os.path.join(self._assets_path, 'weights')
        self.model.wrn38_checkpoint = os.path.join(self.model.weights_path,
            'wider_resnet38.pth.tar')
        self.model.wrn41_checkpoint = os.path.join(self.model.weights_path,
            'wider_resnet41_cornflower_sunfish.pth')
        self.model.x71_checkpoint = os.path.join(self.model.weights_path, 'aligned_xception71.pth')
        self.model.hrnet_checkpoint = os.path.join(self.model.weights_path,
            'hrnetv2_w48_imagenet_pretrained.pth')
        self.model.snapshot = os.path.join(self.model.weights_path, 'satellite_2021.pth')
        weights_path: str = os.path.join(self._assets_path, 'weights')
        self.model.wrn38_checkpoint = os.path.join(weights_path, 'wider_resnet38.pth.tar')
        self.model.wrn41_checkpoint = os.path.join(weights_path,
            'wider_resnet41_cornflower_sunfish.pth')
        self.model.x71_checkpoint = os.path.join(weights_path, 'aligned_xception71.pth')
        self.model.hrnet_checkpoint = os.path.join(weights_path,
            'hrnetv2_w48_imagenet_pretrained.pth')
        self.model.snapshot = os.path.join(weights_path, 'satellite_2021.pth')

        self.dataset.cityscapes_dir = os.path.join(assets, 'data', 'Cityscapes')
        self.dataset.cityscapes_customcoarse = os.path.join(assets,
            *'data Cityscapes autolabelled'.split())
        self.dataset.centroid_root = os.path.join(assets, 'uniform_centroids')
        self.dataset.cityscapes_aug_dir = os.path.join(assets, 'data', 'Mapillary', 'data')
        self.dataset.satellite_dir = os.path.join(assets, 'satellite')

    # result_dir: str = None
    result_dir: str = None
    # snapshot: str = None
    # hrnet_checkpoint: str = None
    quiet: bool = None
    best_record: dict[str, int] = None
    amp_opt_level: str = None
    amsgrad: bool = None
    arch: str = None
    batch_weighting: bool = None
    border_window: int = None
    boundary_path: Optional[str] = None
    brt_aug: bool = None
    calc_metrics: bool = None
    distributed: bool = None
    default_scale: float = None
    deterministic: bool = None
    do_flip: bool = None
    dump_all_images: bool = None
    dump_assets: bool = None
    dump_augmentation_images: bool = None
    dump_for_auto_labelling: bool = None
    dump_topn_all: bool = None
    dump_topn: int = None
    epoch: int = None
    exp: str = None
    freeze_trunk: bool = None
    global_rank: int = None
    city_info_path: Optional[str] = None
    hardnm: int = None
    local_rank: int = None
    log_msinf_to_tb: bool = None
    lr: float = None
    map_crop_val: bool = None
    max_cu_epoch: int = None
    max_epoch = None
    momentum: float = None
    multi_scale_inference: bool = None
    ngpu = None
    num_workers: int = None
    ocr_aux_loss_rmi: bool = None
    old_data = None
    only_coarse: bool = None
    poly_exp: float = None
    poly_step: int = None
    reduce_border_epoch = None
    repoly: float = None
    rescale: float = None
    restore_net: bool = None
    restore_optimizer: bool = None
    resume: str = None
    supervised_mscale_loss_wt: Optional[float] = None
    scf: bool = None
    start_epoch: int = None
    strictborderclass = None
    summary: bool = None
    syncbn: bool = None
    tau_factor: float = None
    tile2net: bool = None
    train_mode: bool = None
    trial: Optional[int] = None
    trunk: str = None
    val_freq: int = None
    weight_decay: float = None
    wt_bound: float = None
    world_size: int = None

    immutable: False

    dump_percent: int = None

    # torch_version = torch_version_float()
    interactive: bool = False
    debug: bool = False

    @cached_property
    def torch_version(self):
        return torch_version_float()

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for key, value in self.__dict__.items():
            yield f'--{key}', value


    # noinspection PyMissingConstructor
    def __init__(self, **kwargs):
        logger.debug('Namespace.__init__')
        # parse nested attributes
        class SetAttr(NamedTuple):
            obj: object
            name: str
            value: Any


        # noinspection PyTypeChecker
        stack = list(map(SetAttr, itertools.repeat(self), kwargs.keys(), kwargs.values()))
        while stack:
            struct = stack.pop()
            if struct.value is None:
                continue
            if '.' in struct.name:
                name, rest = struct.name.split('.', maxsplit=1)
                obj = getattr(struct.obj, name)
                stack.append(SetAttr(obj, rest, struct.value))
            else:
                setattr(*struct)

        project = None
        if (
            not self.interactive
            and not sys.stdin.isatty()
            and (text := sys.stdin.read()) != ''
        ):
            # case: piped
            logger.debug('Inference piped; reading stdin')
            try:
                project = json.loads(text)
            except json.JSONDecodeError as e:
                logger.error(f'Could not parse JSON from stdin: {e}')
                logger.error(f'JSON: {text}')
                raise
            city_info_path = project['tiles']['info']
            with open(city_info_path) as f:
                city_info = json.load(f)
        else:
            # case: unpiped
            logger.debug('Inference unpiped; reading from args')
            logger.debug(f'{self.city_info_path=}')
            if self.city_info_path is not None:
                with open(self.city_info_path) as f:
                    city_info = json.load(f)
                project = city_info['project']

        for key, value in city_info.items():
            if key not in self.__dict__:
                continue
            setattr(self, key, value)


        #   get project structure if piped
        if self.debug:
            logger.setLevel('DEBUG')

        # if project has been determined, get from it; otherwise raise where not defined
        if project is not None:
            if not self.model.snapshot:
                self.model.snapshot = toolz.get_in(
                    'assets weights satellite_2021'.split(),
                    project, no_default=True
                )
            if not self.model.hrnet_checkpoint:
                self.model.hrnet_checkpoint = toolz.get_in(
                    'assets weights hrnetv2_w48_imagenet_pretrained'.split(),
                    project, no_default=True
                )
            if not self.model.weights_path:
                self.model.weights_path = self.model.hrnet_checkpoint.rpartition(os.sep)[0]
            if not self.result_dir:
                self.result_dir = project['segmentation']
            if not self.dataset.name:
                self.dataset.name = project['name']
            if not self.eval_folder:
                self.eval_folder = project['tiles']['stitched']
            if not self.city_info_path:
                self.city_info_path = project['tiles']['info']
        else:
            logger.debug(f'project is None')

        if not self.eval_folder:
            raise ValueError('eval_folder must be set')
        else:
            logger.debug(f'{self.eval_folder=}')

        if not self.result_dir:
            raise ValueError('result_dir must be set')
        if not self.model.snapshot:
            self.model.snapshot = (
                Project.resources.assets.weights.satellite_2021.__fspath__()
            )
            logger.info(
                f'No snapshot specified, using default: {self.model.snapshot}'
            )
        if not self.model.hrnet_checkpoint:
            self.model.hrnet_checkpoint = (
                Project.resources.assets.weights.hrnetv2_w48_imagenet_pretrained.__fspath__()
            )
            logger.info(
                f'No hrnet_checkpoint specified, using default: {self.model.hrnet_checkpoint}'
            )
        if not self.city_info_path:
            raise ValueError('city_info_path must be set')




        self.__dict__.pop('_functions_stack')
        self.update_cfg()

    _mutable = 'epoch dataset_inst'.split()

    def keys(self):
        # skip '_functions' and attrdicts when performing **args unpacking
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                continue
            if key == '_functions':
                continue
            yield key

    @cached_property
    def cfg_args(self):
        class Equivalent:
            left: AttrDict
            left_stack: list[str]
            right: Namespace | AttrDict
            right_stack: list[str]

            def __init__(
                self,
                left: AttrDict,
                right: Namespace | AttrDict,
                left_stack: list[str] = None,
                right_stack: list[str] = None,
            ):
                self.left = left
                self.right = right
                self.left_stack = left_stack or []
                self.right_stack = right_stack or []

            def __iter__(self):
                for left_key, left in self.left.items():
                    right_key = left_key.lower()
                    right = getattr(self.right, right_key)
                    left_stack = self.left_stack + [left_key]
                    right_stack = self.right_stack + [right_key]

                    if isinstance(left, AttrDict):
                        yield from Equivalent(left, right, left_stack, right_stack)
                    else:
                        yield (
                            '.'.join(left_stack),
                            '.'.join(right_stack),
                        )


        res = dict(Equivalent(cfg, self))

        return res

    @cached_property
    def args_cfg(self):
        return {v: k for k, v in self.cfg_args.items()}

    def __repr__(self):
        return 'args'

    def update_cfg(self):
        for lattr, rattr in self.args_cfg.items():
            obj = self
            while '.' in lattr:
                name, lattr = lattr.split('.', maxsplit=1)
                obj = getattr(obj, name)
            value = getattr(obj, lattr)

            obj = cfg
            while '.' in rattr:
                name, rattr = rattr.split('.', maxsplit=1)
                obj = getattr(obj, name)
            setattr(obj, rattr, value)

    @classmethod
    def wrap(cls, func):
        @functools.wraps(func)
        def wrapper(args: argh.ArghNamespace, **kwargs):
            namespace = cls(**args.__dict__)
            return func(namespace, **kwargs)

        return wrapper

# load the values of config into args
key: str
left: Any | AttrDict
not_found: list[str] = []
mismatch: list[str] = []


class Equivalent(NamedTuple):
    left: AttrDict
    left_name: str
    right: Type['Namespace'] | AttrDict
    right_name: str


E = Equivalent

equivalents: list[Equivalent] = [E(cfg, '__C', Namespace, 'args')]

while equivalents:
    equivalent = equivalents.pop()
    for left_key, left in equivalent.left.items():
        right_key = left_key.lower()
        left_name = f'{equivalent.left_name}.{left_key}'
        right_name = f'{equivalent.right_name}.{right_key}'

        if not (
                right_key in equivalent.right.__dict__
                or right_key in equivalent.right.__class__.__dict__
        ):
            not_found.append(
                f'{left_name} not found in \n\t{right_name}'
            )
            continue

        right = getattr(equivalent.right, right_key)
        if isinstance(right, (property, cached_property)):
            continue

        if isinstance(left, AttrDict):
            pipe(
                E(left, left_name, right, right_name),
                equivalents.append
            )
        else:
            obj = getattr(equivalent.right, right_key)
            if hasattr(obj, '__get__'):
                continue
            setattr(equivalent.right, right_key, left)

for v in not_found:
    logger.error(v)
for v in mismatch:
    logger.error(v)
assert (
        not not_found
        and not mismatch
)
