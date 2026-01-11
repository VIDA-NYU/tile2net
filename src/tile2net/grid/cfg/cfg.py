from __future__ import absolute_import, annotations, division, print_function, unicode_literals
from functools import cached_property

import argparse
import functools
import hashlib
import io
import json
import math
import multiprocessing
import os
import re
from collections import deque, UserDict
from pathlib import Path
from types import MappingProxyType
from typing import *
from typing import TypeVar, Callable

import torch
from toolz.curried import *

from tile2net.grid.cfg import cmdline
from tile2net.grid.cfg.colormap import ColorMap

if False:
    from ..grid import Grid

T = TypeVar('T')


class metaclass(type, metaclass=type):
    def __getitem__(self, item: T) -> T:
        return self

    def __call__(self, *args, **kwargs):
        return args[0]


class cached(metaclass=metaclass):
    class classmethod:
        __wrapped__: Callable[..., T]

        def __init__(
                self,
                func: Callable[..., T]
        ):
            functools.update_wrapper(self, func)

        def __set_name__(self, owner, name):
            self.__name__ = name

        def __get__(
                self,
                instance,
                owner
        ):
            result = self.__wrapped__(owner)
            setattr(owner, self.__name__, result)
            return result


class _Default(UserDict):
    data: dict[type[Cfg], Cfg]

    def __get__(
            self,
            instance,
            owner: type[Cfg]
    ):
        cache = self.data
        if owner not in cache:
            cache[owner] = owner.from_defaults()
        return cache[owner]


class Group:
    def __init__(
            self,
            name: str,
            order: float
    ):
        self.name = name
        self.order = order

    def __call__(
            self,
            arg: Union[cmdline.property, float, int]
    ):
        # case 1: used as a factory with a specific order, e.g., @basic(0)
        if isinstance(arg, (int, float)):
            def wrapper(prop: cmdline.property):
                # bypass nested.__setattr__ to avoid runtime config lookups
                object.__setattr__(prop, 'group', self)
                object.__setattr__(prop, 'group_order', arg)
                return prop

            return wrapper

        # case 2: used as a standard decorator, e.g., @basic
        prop = arg
        object.__setattr__(prop, 'group', self)
        object.__setattr__(prop, 'group_order', None)
        return prop


basic = Group('Basic Options', 0)
outputs = Group('Output Options', 1)
architecture = Group('Architecture Options', 2)


class Options(cmdline.Namespace):

    @cmdline.property
    def test_mode(self) -> bool:
        """
        Minimum testing to verify nothing failed,
        runs code for 1 epoch of train and val
        """
        return False

    @cmdline.property
    def init_decoder(self) -> bool:
        """
        Initialize decoder with kaiming normal
        """
        return False

    @cmdline.property
    def torch_version(self) -> float:
        """
        Parsed major.minor torch version as float, e.g. 1.13
        """
        version_str = torch.__version__
        version_re = re.search(r'^([0-9]+\.[0-9]+)', version_str)
        if version_re:
            version = float(version_re.group(1))
            # logger.debug(f'Torch version: {version}, {version_str}')
        else:
            version = 1.0
            # logger.debug(f'Can\'t parse torch version ({version}), assuming {version}')
        return version


class Train(cmdline.Namespace):
    @cmdline.property
    def batch_size(self) -> int:
        """
        Batch size per GPU for the training run
        """
        return 2

    @cmdline.property
    def random_brightness_shift_value(self) -> int:
        """
        Amount of brightness shift to apply randomly during training
        """
        return 10

    @cmdline.property
    def fp16(self) -> bool:
        """
        Use half-precision (fp16) training
        """
        return False


class Dataset(cmdline.Namespace):
    @cmdline.property
    def translate_aug_fix(self) -> bool:
        """
        Use fixed translation augmentation
        """
        return False

    @cmdline.property
    def name(self) -> str:
        """
        Name of the dataset
        """

    @cmdline.property
    def crop_size(self) -> tuple[int, int]:
        """
        Training crop size: either scalar or h,w
        """
        return 1024, 1024

    @cmdline.property
    def centroid_root(self) -> str:
        """
        Path to precomputed uniform sampling centroids
        """

    @cmdline.property
    def satellite_dir(self) -> str:
        """
        Path to satellite imagery directory
        """

    @cmdline.property
    def mean(self) -> list[float]:
        """
        RGB mean for normalization
        """
        return [0.485, 0.456, 0.406]

    @cmdline.property
    def std(self) -> list[float]:
        """
        RGB std for normalization
        """
        return [0.229, 0.224, 0.225]

    @cmdline.property
    def num_classes(self) -> int:
        """
        Number of classes in dataset
        """
        return 4
        # return 0

    @cmdline.property
    def ignore_label(self) -> int:
        """
        Label to ignore during loss computation
        """
        return -1

    @cmdline.property
    def dump_images(self) -> bool:
        """
        Dump input images and predicted masks
        """
        return False

    @cmdline.property
    def class_uniform_pct(self) -> float:
        """
        Percentage of uniform class sampling
        """
        return 0.5

    @cmdline.property
    def class_uniform_tile(self) -> int:
        """
        Tile size for class uniform sampling
        """
        return 512

    @cmdline.property
    def coarse_boost_classes(self) -> list:
        """
        Classes to boost using coarse annotations
        """

    @cmdline.property
    def colorize_mask_fn(self) -> str:
        """
        Function for colorizing segmentation masks
        """

    @cmdline.property
    def mask_out_cityscapes(self) -> bool:
        """
        Mask out cityscapes validation labels
        """
        return False

    @cmdline.property
    def cv(self) -> int:
        """
        Cross-validation split ID
        """
        return 0

    @cmdline.property
    def lanczos_scales(self) -> bool:
        """
        Enable lanczos-based scaling
        """
        return False

    @cmdline.property
    def mapillary_crop_val(self) -> bool:
        """
        Use center crop for mapillary validation
        """
        return False


class Stage1(cmdline.Namespace):

    @cmdline.property
    def num_modules(self) -> int:
        return 1

    @cmdline.property
    def num_ranches(self) -> int:
        return 1

    @cmdline.property
    def block(self) -> str:
        return 'BOTTLENECK'

    @cmdline.property
    def num_blocks(self) -> list[int]:
        return [4]

    @cmdline.property
    def num_channels(self) -> list[int]:
        return [64]

    @cmdline.property
    def fuse_method(self) -> str:
        return 'SUM'


class Stage2(cmdline.Namespace):

    @cmdline.property
    def num_modules(self) -> int:
        return 1

    @cmdline.property
    def num_branches(self) -> int:
        return 2

    @cmdline.property
    def block(self) -> str:
        return 'BASIC'

    @cmdline.property
    def num_blocks(self) -> list[int]:
        return [4, 4]

    @cmdline.property
    def num_channels(self) -> list[int]:
        return [48, 96]

    @cmdline.property
    def fuse_method(self) -> str:
        return 'SUM'


class Stage3(cmdline.Namespace):

    @cmdline.property
    def num_modules(self) -> int:
        return 4

    @cmdline.property
    def num_branches(self) -> int:
        return 3

    @cmdline.property
    def block(self) -> str:
        return 'BASIC'

    @cmdline.property
    def num_blocks(self) -> list[int]:
        return [4, 4, 4]

    @cmdline.property
    def num_channels(self) -> list[int]:
        return [48, 96, 192]

    @cmdline.property
    def fuse_method(self) -> str:
        return 'SUM'


class Stage4(cmdline.Namespace):

    @cmdline.property
    def num_modules(self) -> int:
        return 3

    @cmdline.property
    def num_branches(self) -> int:
        return 4

    @cmdline.property
    def block(self) -> str:
        return 'BASIC'

    @cmdline.property
    def num_blocks(self) -> list[int]:
        return [4, 4, 4, 4]

    @cmdline.property
    def num_channels(self) -> list[int]:
        return [48, 96, 192, 384]

    @cmdline.property
    def fuse_method(self) -> str:
        return 'SUM'


class OcrExtra(cmdline.Namespace):

    @cmdline.property
    def final_conv_kernel(self) -> int:
        return 1

    @Stage1
    def stage1(self) -> Stage1:
        ...

    @Stage2
    def stage2(self) -> Stage2:
        ...

    @Stage3
    def stage3(self) -> Stage3:
        ...

    @Stage4
    def stage4(self) -> Stage4:
        ...


class OCR(cmdline.Namespace):

    @cmdline.property
    def mid_channels(self) -> int:
        """
        Number of mid-level channels in OCR head
        """
        return 512

    @cmdline.property
    def key_channels(self) -> int:
        """
        Number of key channels in OCR head
        """
        return 256


class Model(cmdline.Namespace):

    @cmdline.property
    def lr_scheduler(self) -> str:
        """
        Learning rate scheduler
        """
        return 'poly'

    @cmdline.property
    def optimizer(self) -> str:
        """
        Optimizer type
        """
        return 'sgd'

    @cmdline.property
    def scale_min(self) -> float:
        """
        Dynamically scale training images down to this size
        """
        return 0.5

    @cmdline.property
    def scale_max(self) -> float:
        """
        Dynamically scale training images up to this size
        """
        return 2.0

    @cmdline.property
    def color_aug(self) -> float:
        """
        Color augmentation intensity
        """
        return 0.25

    @cmdline.property
    def gblur(self) -> bool:
        """
        Enable Gaussian blur
        """
        return False

    @cmdline.property
    def bblur(self) -> bool:
        """
        Enable box blur
        """
        return True

    # bblur.add_options(long='--no_bblur')

    @cmdline.property
    def full_crop_modeling(self) -> bool:
        """
        Use full image crops for modeling
        """
        return False

    @cmdline.property
    def eval(self) -> str:
        """
        Just run evaluation, can be set to val or trn, test or folder
        """
        return 'test'

    @cmdline.property
    def pre_size(self) -> str:
        """
        Resize input to this before cropping
        """

    @cmdline.property
    def rand_augment(self) -> list[int]:
        """
        RandAugment setting: set to N,M
        """

    @cmdline.property
    def grad_ckpt(self) -> bool:
        """
        Use gradient checkpointing
        """
        return False

    @cmdline.property
    def rmi_loss(self) -> bool:
        """
        Enable RMI loss
        """
        return False

    @cmdline.property
    def img_wt_loss(self) -> bool:
        """
        Enable per-pixel image-weighted loss
        """
        return True

    # img_wt_loss.add_options(long='--no_img_wt_loss')

    @cmdline.property
    def bn(self) -> str:
        """
        BatchNorm variant to use
        """
        return 'regularnorm'

    # @cmdline.property
    # def bnfunc(self) -> str:
    #     """
    #     BatchNorm function override
    #     """

    @cached_property
    def bnfunc(self):
        return torch.nn.BatchNorm2d

    # @cmdline.property
    # def mscale(self) -> bool:
    #     """
    #     Use multi-scale input during training
    #     """
    #     self._cfg.arch.
    #     return False
    @cached_property
    def mscale(self) -> bool:
        return (
            'mscale' in self._cfg.arch
            or 'attnscale' in self._cfg.arch
        )

    @cmdline.property
    def three_scale(self) -> bool:
        """
        Use three-scale inference
        """
        return False

    @cmdline.property
    def alt_two_scale(self) -> bool:
        """
        Use alternative two-scale inference
        """
        return False

    @cmdline.property
    def align_corners(self) -> bool:
        """
        Use align_corners=True in interpolation
        """
        return False

    @cmdline.property
    def extra_scales(self) -> list[float]:
        return [0.5, 1.5]

    @cmdline.property
    def n_scales(self) -> list[float]:
        """
        Number of scales for multi-scale inference, e.g. '1,2,3'
        """

    @cmdline.property
    def mscale_lo_scale(self) -> float:
        """
        Low resolution scale for multi-scale training
        """
        return 0.5

    @cmdline.property
    def ocr_aspp(self) -> bool:
        """
        Enable OCR+ASPP hybrid module
        """
        return False

    @cmdline.property
    def segattn_bot_ch(self) -> int:
        """
        Bottleneck channels in segmentation attention module
        """
        return 256

    @cmdline.property
    def aspp_bot_ch(self) -> int:
        """
        Bottleneck channels in ASPP module
        """
        return 256

    @cmdline.property
    def mscale_cat_scale_flt(self) -> bool:
        """
        Concatenate scale features as float
        """
        return False

    @cmdline.property
    def mscale_inner_3x3(self) -> bool:
        """
        Use 3x3 conv in mscale inner branch
        """
        return True

    # mscale_inner_3x3.add_options(long='--no_mscale_inner_3x3')

    @cmdline.property
    def mscale_dropout(self) -> bool:
        """
        Enable dropout in mscale module
        """
        return False

    @cmdline.property
    def mscale_oldarch(self) -> bool:
        """
        Use old mscale architecture
        """
        return False

    @cmdline.property
    def mscale_init(self) -> float:
        """
        Initial scale weight for mscale fusion
        """
        return 0.5

    @cmdline.property
    def attnscale_bn_head(self) -> bool:
        """
        Enable BN head in attention scale
        """
        return False

    @cmdline.property
    def wrn38_checkpoint(self) -> str:
        """
        Path to WRN-38 checkpoint
        """

    @cmdline.property
    def wrn41_checkpoint(self) -> str:
        """
        Path to WRN-41 checkpoint
        """

    @cmdline.property
    def x71_checkpoint(self) -> str:
        """
        Path to Xception-71 checkpoint
        """

    @cmdline.property
    def hrnet_checkpoint(self) -> str:
        """
        Path to HRNet checkpoint
        """
        from tile2net.grid.grid.static import Static
        return Static.hrnet_checkpoint

    @cmdline.property
    def snapshot(self) -> str:
        """
        Path to the model snapshot
        """
        from tile2net.grid.grid.static import Static
        return Static.snapshot

    @cmdline.property
    def trial(self) -> int:
        """
        Trial number for experiments
        """

    @cmdline.property
    def val_freq(self) -> int:
        """
        How often (in epochs) to run validation
        """
        return 1

    @OcrExtra
    def ocr_extra(self) -> OcrExtra:
        ...

    @OCR
    def ocr(self) -> OCR:
        ...


class Download(cmdline.Namespace):
    @cmdline.property
    def force(self):
        return False

    @basic
    @cmdline.property
    def only(self) -> bool:
        """Do not perform any tasks past the downloading step."""
        return False


class Segmentation(
    cmdline.Namespace
):
    """Segmenation configuration namespace."""

    @cmdline.property
    def dimension(self) -> int:
        """
        Dimension of each segmentation mask input into vectorization.
        """
        return 1024

    @basic
    @cmdline.property
    def length(self) -> int:
        """
        Length, in input images, of each seg-tile.
        A length of 10, for example, means each vec-tile
        is 10 seg-tiles long.
        """

    @cmdline.property
    def scale(self) -> int:
        """
        XYZ scale of each segmentation mask.
        Scale=17 means the input segmentation mask comprises the same
        area as a zoom=17 slippy tile of dimension=256.
        """

    @cmdline.property
    def fill(self) -> bool:
        return True

    @cmdline.property
    def pad(self) -> int:
        """Number of seg-tiles to pad each vec-tile by."""
        return 1

    @cmdline.property
    def prefetch_factor(self) -> int:
        """
        Number of batches to prefetch
        """
        return 4

    @cmdline.property
    def persistent_workers(self) -> bool:
        """
        Keep data loader workers alive
        """
        # return False
        return True

    @cmdline.property
    def pred(self) -> bool:
        """Save predictions to file at the respective scale."""
        return False

    @cmdline.property
    def prob(self) -> bool:
        """Save per-class probabilities to file at the respective scale."""
        return False

    @cmdline.property
    def colorized(self) -> bool:
        """Save colorized segmentation masks to file at the respective scale."""
        return False

    @cmdline.property
    def intensity(self) -> bool:
        """Save intensity visualization to file at the respective scale."""
        return False

    @cmdline.property
    def sidebyside(self) -> bool:
        """Save side-by-side input and colorized segmentation masks to file at the respective scale."""
        return False

    @cmdline.property
    def overlay(self) -> bool:
        """Save overlay visualization to file at the respective scale."""
        return False

    @cmdline.property
    def error(self) -> bool:
        """Save error visualization to file at the respective scale."""
        return False

    @cmdline.property
    def soft(self) -> bool:
        """Save soft segmentation masks (color multiplied by probability) to file at the respective scale."""
        return False

    @cmdline.property
    def static(self) -> bool:
        """Save static imagery to file at the respective scale."""
        return False

    @cmdline.property
    def polygons(self) -> bool:
        """Save vectorized polygons to file at the respective scale."""
        return False

    @cmdline.property
    def network(self) -> bool:
        """Save vectorized network to file at the respective scale."""
        return False

    @basic
    @cmdline.property
    def only(self) -> bool:
        """Do not perform any tasks past the segmentation step."""
        return False


class Vectorization(
    cmdline.Namespace
):
    """Vectorization configuration namespace."""

    @cmdline.property
    def dimension(self) -> int:
        """
        Dimension of each concatenated segmentation mask before it is
        put into vectorization. The amount of pixels wide each image
        is before geometries are extracted.
        """

    @basic
    @cmdline.property
    def length(self) -> int:
        """
        Length, in seg-tiles, of each vec-tile.
        A length of 10, for example, means each vec-tile
        is 10 seg-tiles long.
        """
        return 1

    length.add_options(short='-v')

    @cmdline.property
    def scale(self) -> int:
        """
        XYZ scale of each segmentation mask.
        Scale=17 means the input segmentation mask comprises the same
        area as a zoom=17 slippy tile of dimension=256.
        """

    @cmdline.property
    def pad(self) -> int:
        """Number of in-tiles to pad each seg-tile by."""
        return 1

    @basic
    @cmdline.property
    def batch_size(self):
        """
        Batch size per GPU for the validation run
        """
        return 1

    @cmdline.property
    def persistent_workers(self) -> bool:
        """
        Keep data loader workers alive
        """
        return False

    @cmdline.property
    def num_workers(self):
        return multiprocessing.cpu_count()

    @cmdline.property
    def num_loaders(self):
        return 4

    @cmdline.property
    def pred(self) -> bool:
        """Save predictions to file at the respective scale."""
        return False

    @cmdline.property
    def prob(self) -> bool:
        """Save per-class probabilities to file at the respective scale."""
        return False

    @cmdline.property
    def colorized(self) -> bool:
        """Save colorized segmentation masks to file at the respective scale."""
        return False

    @cmdline.property
    def intensity(self) -> bool:
        """Save intensity visualization to file at the respective scale."""
        return False

    @cmdline.property
    def sidebyside(self) -> bool:
        """Save side-by-side input and colorized segmentation masks to file at the respective scale."""
        return False

    @cmdline.property
    def overlay(self) -> bool:
        """Save overlay visualization to file at the respective scale."""
        return False

    @cmdline.property
    def error(self) -> bool:
        """Save error visualization to file at the respective scale."""
        return False

    @cmdline.property
    def soft(self) -> bool:
        """Save soft segmentation masks (color multiplied by probability) to file at the respective scale."""
        return False

    @cmdline.property
    def static(self) -> bool:
        """Save static imagery to file at the respective scale."""
        return False

    @cmdline.property
    def polygons(self) -> bool:
        """Save vectorized polygons to file at the respective scale."""
        return False

    @cmdline.property
    def network(self) -> bool:
        """Save vectorized network to file at the respective scale."""
        return False


class Loss(cmdline.Namespace):

    @cmdline.property
    def ocr_alpha(self) -> float:
        """
        Weight for OCR auxiliary loss
        """
        return 0.4

    @cmdline.property
    def ocr_aux_rmi(self) -> bool:
        """
        Use RMI loss for the OCR auxiliary branch
        """
        return False

    @cmdline.property
    def supervised_mscale_wt(self) -> float:
        """
        Weight for supervising multi-scale predictions
        """
        return 0.0


class Line(cmdline.Namespace):

    @cmdline.property
    def force(self):
        return False

    @cmdline.property
    def concat(self) -> bool:
        """Concatenate the lines from each tile into a single vector."""
        return True

    @cmdline.property
    def preview(self) -> int:
        """Maximum dimension of the lines preview"""
        return 3072


class Curb(cmdline.Namespace):
    @cmdline.property
    def distance(self) -> float:
        return 3.


class Polygon(cmdline.Namespace):

    @cmdline.property
    def force(self):
        return False

    @cmdline.property
    def max_hole_area(self) -> float | dict[str, float]:
        return dict(
            road=30,
            crosswalk=15
        )

    @cmdline.property
    def grid_size(self) -> float:
        ...

    @cmdline.property
    def min_polygon_area(self) -> float | dict[str, float]:
        return 20

    @cmdline.property
    def convexity(self) -> float | dict[str, float]:
        return 0.8

    @cmdline.property
    def simplify(self) -> float | dict[str, float]:
        # return 5.
        return 0.8

    @cmdline.property
    def z_order(self) -> dict[str, int]:
        """Map label names to z-order values."""
        return dict(
            # sidewalk=0,
            # road=1,
            road=0,
            sidewalk=1,
            crosswalk=2,
        )

    @cmdline.property
    def borders(self) -> list[str]:
        """
        Feature which are included not as pedestrian networks
        but as borders to them.
        """
        return ['road']

    @cmdline.property
    def concat(self) -> bool:
        """Concatenate the polygons from each tile into a single vector."""
        return True

    @cmdline.property
    def preview(self) -> int:
        """Maximum dimension of the polygons preview"""
        return 3072

    @cmdline.property
    def thickness(self) -> float:
        """Line thickness for polygon preview"""
        return 1.


class Validation(
    cmdline.Namespace
):
    @cmdline.property
    def batch_size(self):
        """
        Batch size per GPU for the validation run
        """
        return 1

    batch_size.add_options(short='-b')


class Indir(
    cmdline.Namespace
):
    @cmdline.property
    def path(self) -> Optional[str]:
        """
        Path to the input directory containing imagery tiles. The path
        should implicate the format of the files, containing the xtile,
        ytile, and extension, and possibly the zoom level.

        For example, `path/to/tiles/z/x/y.png` tells Tile2Net we are
        working with PNG files stored under the format
        /{zoom}/{xtile}/{ytile}.png, and searches for files as such.

        If the user passes `path/to/tiles/x_y.png`, Tile2Net will look
        for files with names `{xtile}_{ytile}.png` in the `tiles` folder.

        If not specified, the tiles will be downloaded from a remote source.
        """

    path.add_options(short='-i')

    @cmdline.property
    def name(self) -> Optional[str]:
        ...


class Cfg(
    UserDict,
    cmdline.Namespace
):
    grid = None
    grid: Grid
    instance: Grid = None
    owner: Type[Grid] = None
    __name__ = ''
    _active = True

    _default = _Default()

    # _default: Self = None
    _context: Self = None
    _backup: Self = None

    def _lookup(self, trace: str):
        # try local
        if (
                self is not self._default
                and trace in self
        ):
            return self[trace]
        # try context
        if (
                self._context
                and trace in self._context
        ):
            return self._context[trace]
        # try default
        if (
                self._default
                and trace in self._default
        ):
            return self._default[trace]
        msg = f'No cached config value for {trace!r}'
        raise KeyError(msg)
        # try:
        #     return getattr(self, trace)
        # except AttributeError as e:
        #     msg = f'No cached config value for {self._trace!r}'
        #     raise KeyError(msg) from e

    def _evict(self, trace: str):
        """
        Evict a cached value from the configuration.
        """
        if trace in self._cfg:
            del self._cfg[trace]

    def _get(
            self,
            instance: Grid,
            owner: type[Grid]
    ) -> Self:
        if instance is None:
            result = self
        elif isinstance(instance, Cfg):
            return self
        elif self._trace in instance.__dict__:
            result = instance.__dict__[self._trace]
        else:
            result = self.__class__()
            instance.__dict__[self._trace] = result
        result.__name__ = self.__name__
        result.grid = instance
        result.Grid = owner
        result.instance = instance
        result.owner = owner
        assert result is not None
        return result

    locals().update(__get__=_get)

    def __init__(self, dict=None, /, **kwargs):
        if callable(dict):
            UserDict.__init__(self, **kwargs)
        else:
            UserDict.__init__(self, dict, **kwargs)

        # if (
        #         args
        #         and callable(args[0])
        # ):
        #     # super().__init__(*args[1:], **kwargs)
        #     self.update(self._trace2default)
        # else:
        #     UserDict.__init__(self)
        #     # super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        lines = (f'  "{k}": {v!r}' for k, v in self.items())
        return "{\n" + ",\n".join(lines) + "\n}"

    @Options
    def options(self):
        ...

    @Train
    def train(self):
        """Namespace for arguments pertaining to training."""

    @Dataset
    def dataset(self):
        """Namespace for arguments pertaining to dataset handling."""

    @Loss
    def loss(self):
        """Namespace for arguments pertaining to loss computation."""

    @Model
    def model(self):
        """Namespace for arguments pertaining to model architecture and training."""

    @Download
    def download(self):
        """Namespace for arguments pertaining to downloading static imagery tiles."""

    @Segmentation
    def segmentation(self):
        """Namespace for arguments pertaining to segmentation."""

    @Vectorization
    def vectorization(self):
        """Namespace for arguments pertaining to vectorization."""

    @Polygon
    def polygon(self):
        """Namespace for arguments pertaining to polygon vectorization."""

    @Curb
    def curb(self):
        """Namespace for arguments pertaining to curb vectorization."""

    @Indir
    def indir(self):
        """Namespace for arguments pertaining to input directory of imagery tiles."""

    @Line
    def line(self):
        """Namespace for arguments pertaining to line vectorization."""

    @Validation
    def validation(self):
        """Namespace for arguments pertaining to validation."""

    @cmdline.property
    def static(self) -> bool:
        """Save static imagery to file at the respective scale."""
        return True

    @cmdline.property
    def inference(self) -> bool:
        """
        Run inference only, skipping training
        """
        return True

    # inference.add_options(long='--no-inference')

    @basic
    @cmdline.property
    def pred(self) -> bool:
        """Save predictions, aligned with the source imagery, to the output directory."""
        return False

    @cmdline.property
    def prob(self) -> bool:
        """Save per-class probabilities, aligned with the source imagery, to the output directory."""
        return False

    @basic
    @cmdline.property
    def colorized(self) -> bool:
        """Save colorized segmentation masks, aligned with the source imagery, to the output directory."""
        return False

    @cmdline.property
    def intensity(self) -> bool:
        """Save intensity visualization, aligned with the source imagery, to the output directory."""
        return False

    @basic
    @cmdline.property
    def sidebyside(self) -> bool:
        """Save static imagery and colorized segmentation masks side-by-side, aligned with the source imagery,
        to the output directory.
        """
        return False

    @basic
    @cmdline.property
    def overlay(self) -> bool:
        """Save overlay visualization, aligned with the source imagery, to the output directory."""
        return False

    @cmdline.property
    def error(self) -> bool:
        """Save error visualization, aligned with the source imagery, to the output directory."""
        return False

    @basic
    @cmdline.property
    def soft(self) -> bool:
        """Save soft segmentation masks (color multiplied by probability), aligned with the source imagery, to the output directory."""
        return False

    @basic
    @cmdline.property
    def static(self) -> bool:
        """Save static imagery, aligned with the source imagery, to the output directory."""
        return True

    @basic
    @cmdline.property
    def polygons(self) -> bool:
        """Save vectorized polygons, aligned with the source imagery, to the output directory."""
        return True

    @basic
    @cmdline.property
    def network(self) -> bool:
        """Save vectorized network, aligned with the source imagery, to the output directory."""
        return True

    def label2id(self) -> dict[str, int]:
        """
        Mapping from label names to IDs
        """
        return dict(
            sidewalk=0,
            road=1,
            crosswalk=2,
            ignore_label=3
        )

    @basic
    @cmdline.property
    def label2color(self) -> dict[str, str]:
        """
        Mapping from label names to colors
        """
        return dict(
            sidewalk='red',
            road='cyan',
            crosswalk='yellow',
            curb='blue',
            ignore_label='black'
        )

    @functools.cached_property
    def colormap(self) -> ColorMap:
        """
        Build ColorMap from label2id and label2color mappings.
        Maps class IDs to RGB colors for visualization.
        """
        return ColorMap.from_mappings(
            label2color=self.label2color,
            label2id=self.label2id,
        )

    @cmdline.property
    def label2soft_color(self) -> dict[str, str]:
        """
        Mapping from label names to colors for soft segmentation visualization
        """
        return dict(
            sidewalk='red',
            road='blue',
            crosswalk='green',
        )

    @functools.cached_property
    def soft_colormap(self) -> ColorMap:
        """
        Build ColorMap from label2id and label2soft_color mappings.
        Maps class IDs to RGB colors for soft segmentation visualization.
        """
        return ColorMap.from_mappings(
            label2color=self.label2soft_color,
            label2id=self.label2id,
        )

    @basic(1)
    @cmdline.property
    def outdir(self) -> str:
        """
        Path to the output directory; '~/tmp/tile2net' by default.
        Using a relative path such as './cambridge' will create a
        directory in the current working directory.
        """
        return './outdir/z/x_y'

    outdir.add_options(
        short='-o',
    )

    @cmdline.property
    def dump_percent(self) -> int:
        """
        The percentage of segmentation results to save. 100 means all, 0 means none.
        """
        return 0

    @cmdline.property
    def assets_path(self) -> str:
        """"""

    @cmdline.property
    def quiet(self) -> bool:
        """
        Suppress all output
        """
        return False

    quiet.add_options(short='--q')

    @cmdline.property
    def arch(self) -> str:
        """
        Network architecture
        """
        return 'ocrnet.HRNet_Mscale'

    @cmdline.property
    def local_rank(self) -> int:
        """
        Parameter for distributed training
        """
        return 0

    @cmdline.property
    def global_rank(self) -> int:
        """
        Parameter used for distributed training
        """
        return 0

    @cmdline.property
    def world_size(self) -> int:
        """
        Number of processes in distributed training
        """
        return 1

    param = 5

    @cmdline.property
    def trunk(self) -> str:
        """
        Trunk model, can be: hrnetv2 (default), resnet101, resnet50
        """
        return 'hrnetv2'

    @cmdline.property
    def start_epoch(self) -> int:
        """
        Starting epoch for training
        """
        return 0

    @cmdline.property
    def restore_net(self) -> bool:
        """
        Continue training from a checkpoint. Weights, optimizer, schedule are restored.
        """
        return False

    @cmdline.property
    def exp(self) -> str:
        """
        Experiment directory name
        """
        return 'default'

    @cmdline.property
    def syncbn(self) -> bool:
        """
        Use Synchronized Batch Normalization
        """
        return False

    @cmdline.property
    def dump_augmentation_images(self) -> bool:
        """
        Dump Augmented Images for sanity check
        """
        return False

    @cmdline.property
    def multi_scale_inference(self) -> bool:
        """
        Run multi-scale inference
        """
        return True

    @cmdline.property
    def default_scale(self) -> float:
        """
        Default scale to run validation
        """
        return 1.0

    @cmdline.property
    def log_msinf_to_tb(self) -> bool:
        """
        Log multi-scale Inference to Tensorboard
        """
        return False

    @cmdline.property
    def resume(self) -> str:
        """
        Continue training from a checkpoint. Weights, optimizer, schedule are restored.
        """

    @cmdline.property
    def max_epoch(self) -> int:
        """
        Maximum number of epochs for training
        """
        return 300

    @cmdline.property
    def max_cu_epoch(self) -> int:
        """
        Class Uniform Max Epochs
        """
        return 150

    @cmdline.property
    def rescale(self) -> float:
        """
        Warm Restarts new learning rate ratio compared to original learning rate
        """
        return 1.0

    @cmdline.property
    def ocr_aux_loss_rmi(self) -> bool:
        """
        Allow RMI for auxiliary loss
        """
        return False

    @cmdline.property
    def tau_factor(self) -> float:
        """
        Factor for NASA optimization function
        """
        return 1.0

    @cmdline.property
    def tile2net(self) -> bool:
        """
        If true, creates the polygons and lines from the results
        """
        return True

    @cmdline.property
    def boundary_path(self) -> Optional[str]:
        """
        Path to the boundary file shapefile
        """

    @cmdline.property
    def interactive(self) -> bool:
        """
        Tile2net is being run in interactive Python
        """
        return False

    @cmdline.property
    def debug(self) -> bool:
        """
        Enable debug mode
        """
        return False

    debug.add_options(short='-d')

    @cmdline.property
    def local(self) -> bool:
        """
        Run in local mode
        """
        return False

    @cmdline.property
    def remote(self) -> bool:
        """
        Run in remote mode
        """
        return False

    @cmdline.property
    def lr(self) -> float:
        """"""
        return 0.002

    @cmdline.property
    def old_data(self) -> bool:
        """"""
        return False

    @cmdline.property
    def batch_weighting(self) -> bool:
        """"""
        return True

    batch_weighting.add_options(long='--no_batch_weighting')

    @cmdline.property
    def repoly(self) -> float:
        """"""
        return 1.5

    @cmdline.property
    def amsgrad(self) -> Optional[bool]:
        """"""

    @cmdline.property
    def freeze_trunk(self) -> bool:
        """"""
        return False

    @cmdline.property
    def hardnm(self) -> int:
        """"""
        return 0

    @cmdline.property
    def brt_aug(self) -> bool:
        """"""
        return False

    @cmdline.property
    def poly_exp(self) -> float:
        """"""
        return 2.0

    @cmdline.property
    def poly_step(self) -> int:
        """"""
        return 110

    @cmdline.property
    def weight_decay(self) -> float:
        """"""
        return 1e-4

    @cmdline.property
    def momentum(self) -> float:
        """"""
        return 0.9

    @cmdline.property
    def restore_optimizer(self) -> bool:
        """"""
        return False

    @cmdline.property
    def summary(self) -> bool:
        """"""
        return False

    @cmdline.property
    def calc_metrics(self) -> bool:
        """"""
        return False

    @cmdline.property
    def ngpu(self) -> int:
        """"""
        return torch.cuda.device_count()

    @cmdline.property
    def distributed(self) -> bool:
        """"""
        return False

    @cmdline.property
    def amp_opt_level(self) -> str:
        """"""
        return 'O1'

    @cmdline.property
    def dump_topn_all(self) -> bool:
        """"""
        return True

    dump_topn_all.add_options(long='--no_dump_topn_all')

    @cmdline.property
    def only_coarse(self) -> bool:
        """"""
        return False

    @cmdline.property
    def deterministic(self) -> bool:
        """
        Use deterministic training
        """
        return False

    @cmdline.property
    def dump_topn(self) -> int:
        """
        Dump worst validation images
        """
        return 50

    @cmdline.property
    def dump_assets(self) -> bool:
        """
        Dump interesting assets
        """
        return False

    @cmdline.property
    def dump_all_images(self) -> bool:
        """
        Dump all images, not just a subset
        """
        return False

    @cmdline.property
    def name(self) -> str:
        ...

    name.add_options(short='-n')

    @basic(0)
    @cmdline.property
    def location(self) -> str:
        """
        Textual address or bounding box in coordinates for your batch.
        The address can be for example 'Washington Square Park' and the
        bounding box must be in lat, lon order such as
        '40.729, -73.999, 40.732, -73.995'
        """

    location.add_options(short='-l')

    @basic
    @cmdline.property
    def zoom(self) -> int:
        return 20

    zoom.add_options(short='-z')

    @cmdline.property
    def crs(self) -> int:
        return 4326

    @cmdline.property
    def padding(self) -> bool:
        return True

    padding.add_options(long='--nopadding')

    # @cmdline.property
    # def extension(self) -> str:
    #     return 'png'
    # @cmdline.property
    # def tile_step(self) -> int:
    #     return 1
    #
    # @cmdline.property
    # def stitch_step(self) -> int:
    #     return 4
    #
    # stitch_step.add_options(short='-st')

    @basic(1)
    @cmdline.property
    def source(self) -> Optional[str]:
        ...

    source.add_options(short='-s')

    @cmdline.property
    def border_window(self) -> int:
        """
        Border relaxation count
        """
        return 1

    @cmdline.property
    def do_flip(self) -> bool:
        """
        Enable horizontal flipping in inference
        """
        return False

    @cmdline.property
    def epoch(self) -> int:
        """
        Runtime epoch counter
        """
        return 0

    @cmdline.property
    def load_workers(self) -> int:
        """
        Number of workers for data loading
        """
        return min(os.cpu_count() // 2, 4)

    @cmdline.property
    def compress_workers(self) -> int:
        """Number of workers for file compression"""
        return min(os.cpu_count() // 2, 12)

    @cmdline.property
    def reduce_border_epoch(self) -> int:
        """
        Number of epochs to use before disabling border restriction
        """
        return -1

    @cmdline.property
    def scf(self) -> bool:
        """
        Enable SCF (Scale-Channel Fusion)
        """
        return False

    @cmdline.property
    def strictborderclass(self) -> Optional[str]:
        """
        Comma-separated list of class ids for border relaxation
        """

    @cmdline.property
    def wt_bound(self) -> float:
        """
        Class weighting bound
        """
        return 1.0

    @cmdline.property
    def force(self) -> bool:
        """
        Force overwrite existing output files
        """
        return False

    # todo: cfg._best_record doesn't seem to be used anywhere, but evalmetrics.best_record is, so I commented it out
    # @cmdline.property
    # def _best_record(self) -> dict[str, Union[int, float]]:
    #     return dict(
    #         epoch=-1,
    #         iter=0,
    #         val_loss=1e10,
    #         acc=0,
    #         acc_cls=0,
    #         mean_iu=0,
    #         fwavacc=0
    #     )

    @cmdline.property
    def log_level(self) -> str:
        """
        Logging level for the application
        """
        # DEBUG for now, even for end-user probably better while we are developing
        return 'DEBUG'
        # return 'INFO'

    @classmethod
    def from_defaults(cls) -> Self:
        result = cls()
        result.update(result._trace2default)
        return result

    @classmethod
    def from_parser(cls) -> Self:
        parser = Cfg.parser
        namespace = parser.parse_args()
        result = cls()
        default = cls._default
        update = {
            key: value
            for key, value in namespace.__dict__.items()
            if value != default[key]
        }
        result.update(update)
        return result

    @cached[dict[str, cmdline.property]]
    @cached.classmethod
    def _trace2property(cls) -> dict[str, cmdline.property]:
        active = cls._active
        cls._active = False
        nested = [
            getattr(cls, key)
            for key in cls._nested
        ]
        result: dict[str, cmdline.property] = {
            value._trace: value
            for value in nested
            if isinstance(value, cmdline.property)
        }
        stack: deque[cmdline.Namespace] = deque([
            value
            for value in nested
            if isinstance(value, cmdline.Namespace)
        ])
        while stack:
            namespace = stack.popleft()
            for key, value in namespace._nested.items():
                if isinstance(value, cmdline.property):
                    value: cmdline.property = getattr(namespace, key)
                    key = f'{namespace._trace}.{key}'
                    result[key] = value
                elif isinstance(value, cmdline.Namespace):
                    stack.append(getattr(namespace, key))
                else:
                    raise TypeError(f"Unexpected type {type(value)} in _trace2property")

        cls._active = active
        return result

    @cached[argparse.ArgumentParser]
    @cached.classmethod
    def parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="tile2net",
            # todo: good description in CLI
            # description="Tile2Net",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            usage='%(prog)s [options]'
        )
        used_groups = {
            prop.group
            for prop in cls._trace2property.values()
            if prop.group
        }

        it = sorted(used_groups, key=lambda g: g.order)
        arg_groups = {
            group: parser.add_argument_group(group.name)
            for group in it
        }

        def prop_sort_key(prop: cmdline.property):
            group = prop.group
            primary = group.order if group else 1000
            order = prop.group_order
            is_unordered = not isinstance(order, (int, float))
            if not is_unordered:
                val_order = order
            else:
                val_order = 0
            quaternary = prop.dest
            return primary, is_unordered, val_order, quaternary

        all_props = sorted(cls._trace2property.values(), key=prop_sort_key)
        seen_opts: set[str] = set()

        for prop in all_props:
            if any(opt in seen_opts for opt in prop.posargs):
                raise ValueError(f"Duplicate option detected in {prop.posargs}")

            seen_opts.update(prop.posargs)
            group = prop.group
            if (
                    group is not None
                    and group in arg_groups
            ):
                target = arg_groups[group]
            else:
                target = parser

            target.add_argument(*prop.posargs, **prop.kwargs)

        def get_group_order(action_group):
            if action_group.title == 'positional arguments':
                return -100
            if action_group.title == 'options':
                return 1000

            for g in used_groups:
                if g.name == action_group.title:
                    return g.order

            return 500

        # todo: doesn't seem to work, how do we make the default "options" look nice after "Basic Options"?
        # for group in parser._action_groups:
        #     if group.title == 'options':
        #         group.title = 'Options'
        #         break

        parser._action_groups.sort(key=get_group_order)

        return parser

    @cached[dict[str, cmdline.property]]
    @cached.classmethod
    def _trace2default(cls) -> Mapping[str, cmdline.property]:
        active = cls._active
        cls._active = False
        nested = [
            getattr(cls, key)
            for key in cls._nested
        ]
        result: dict[str, cmdline.property] = {
            value._trace: value.default
            for value in nested
            if isinstance(value, cmdline.property)
        }
        stack: deque[cmdline.Namespace] = deque([
            value
            for value in nested
            if isinstance(value, cmdline.Namespace)
        ])
        while stack:
            namespace = stack.popleft()
            for key, value in namespace._nested.items():
                if isinstance(value, cmdline.property):
                    value: cmdline.property = getattr(namespace, key)
                    result[value._trace] = value.default
                elif isinstance(value, cmdline.Namespace):
                    stack.append(getattr(namespace, key))
                else:
                    raise TypeError(f"Unexpected type {type(value)} in _trace2property")

        cls._active = active
        return MappingProxyType(dict(sorted(result.items())))

    def __call__(self, *args, **kwargs) -> Cfg:
        return self

    def __enter__(self):
        context = Cfg._context
        self._backup = context
        if context is not None:
            result = context.copy()
            result.update(self)
        else:
            result = self.copy()
        Cfg._context = result
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore context first, regardless of exception
        if Cfg._context is self:
            Cfg._context = self._backup

        # let exception propagate naturally by returning False
        # don't re-raise explicitly - that interferes with ExitStack
        if exc_type is not None:
            return False

        # Always let exceptions propagate
        return False

    @property
    def _cfg(self) -> Self:
        return self

    @property
    def _Cfg(self) -> type[Self]:
        return type(self)

    def flatten(self) -> Self:
        merged = type(self)()  # new empty Cfg

        # clear linkage/context on the flattened copy
        merged._default = None
        merged._context = None
        merged._backup = None
        merged._active = self._active
        merged.__name__ = self.__name__

        # collect allowed keys from the default config
        # allowed = set(self._default or ())
        allowed = set(self.from_defaults())

        # staged updates in precedence order
        staging: dict = {}
        if self._default:
            staging.update(self._default)
        if self._context:
            staging.update(self._context)
        if self is not self._default:
            staging.update(self)

        # filter: only keep keys that exist in default
        merged.update({
            k: v
            for k, v in staging.items()
            if k in allowed
        })

        return merged

    def hash(self) -> str:
        # compute a stable digest of the flattened config using only elementary JSON-safe types

        # sentinel to drop disallowed values
        _SKIP = object()

        # accept str, int, float (finite), bool
        def _is_elem(x) -> bool:
            if isinstance(x, bool):
                return True
            if isinstance(x, (str, int)):
                return True
            if isinstance(x, float):
                return math.isfinite(x)
            return False

        # recursively filter to {str: elem|list|dict} and lists of elem|list|dict
        def _sanitize(obj):
            if _is_elem(obj):
                return obj
            if isinstance(obj, Mapping):
                # keep only string keys
                kept = []
                for k, v in obj.items():
                    if not isinstance(k, str):
                        continue
                    sv = _sanitize(v)
                    if sv is not _SKIP:
                        kept.append((k, sv))
                # sort for deterministic serialization
                return {k: v for k, v in sorted(kept, key=lambda kv: kv[0])}
            if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
                out = []
                for el in obj:
                    se = _sanitize(el)
                    if se is not _SKIP:
                        out.append(se)
                return out
            return _SKIP

        # flatten then sanitize
        flat = self.flatten()
        sanitized = _sanitize(flat)

        # deterministic JSON encoding
        payload = json.dumps(
            sanitized,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False,
        )

        # short but strong digest
        result = hashlib.blake2b(payload.encode('utf-8'), digest_size=16).hexdigest()
        return result

    def to_json(
            self,
            path: str | os.PathLike | IO[str] = None,
            *,
            indent: int | None = 2,
            ensure_ascii: bool = False,
    ) -> None:
        if path is None:
            path = './cfg.json'
        # write a sanitized, flattened JSON file
        # flatten first to fix precedence and drop non-default keys
        flat = self.flatten()

        # sentinel for dropped values
        _SKIP = object()

        # accept only elementary JSON-safe scalars
        def _is_elem(x: Any) -> bool:
            if isinstance(x, bool):
                return True
            if isinstance(x, (str, int)):
                return True
            if isinstance(x, float):
                return math.isfinite(x)
            return False

        # recursively sanitize to elem|list|dict with str keys
        def _sanitize(obj: Any):
            if _is_elem(obj):
                return obj
            if isinstance(obj, Mapping):
                kept = []
                for k, v in obj.items():
                    if not isinstance(k, str):
                        continue
                    sv = _sanitize(v)
                    if sv is not _SKIP:
                        kept.append((k, sv))
                return {k: v for k, v in sorted(kept, key=lambda kv: kv[0])}
            if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
                out = []
                for el in obj:
                    se = _sanitize(el)
                    if se is not _SKIP:
                        out.append(se)
                return out
            return _SKIP

        payload = _sanitize(flat)

        # choose sink
        if isinstance(path, (str, os.PathLike, Path)):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', encoding='utf-8') as f:
                json.dump(payload, f, sort_keys=True, separators=(',', ':'), indent=indent, ensure_ascii=ensure_ascii)
                f.write('\n')
        elif isinstance(path, io.TextIOBase) or hasattr(path, 'write'):
            json.dump(payload, path, sort_keys=True, separators=(',', ':'), indent=indent, ensure_ascii=ensure_ascii)
            path.write('\n')
        else:
            raise TypeError(f'Unsupported sink type: {type(path)!r}')

    @classmethod
    def from_json(
            cls,
            src: str | os.PathLike | IO[str] | Mapping[str, Any] = None,
    ) -> Self:
        if src is None:
            src = './cfg.json'
        # load JSON, filter to allowed keys, and build a Cfg overriding defaults
        # load object
        if isinstance(src, Mapping):
            loaded = dict(src)
        elif isinstance(src, (str, os.PathLike, Path)):
            # try as path; if not a file, treat as JSON string
            p = Path(src)
            if p.exists():
                with p.open('r', encoding='utf-8') as f:
                    loaded = json.load(f)
            else:
                loaded = json.loads(str(src))
        elif isinstance(src, io.TextIOBase) or hasattr(src, 'read'):
            loaded = json.load(src)
        else:
            raise TypeError(f'Unsupported source type: {type(src)!r}')

        # only accept flat string keys at the top level
        if not isinstance(loaded, Mapping):
            raise ValueError('Top-level JSON must be an object')
        loaded = {str(k): v for k, v in loaded.items()}

        # establish the set of allowed traces from defaults
        defaults = cls.from_defaults()
        allowed = set(defaults)

        # filter to allowed keys and JSON-safe elementary values, mirroring hash() constraints
        def _is_elem(x: Any) -> bool:
            if isinstance(x, bool):
                return True
            if isinstance(x, (str, int)):
                return True
            if isinstance(x, float):
                return math.isfinite(x)
            return False

        def _sanitize_incoming(v: Any) -> Any:
            # allow elem, list of elem, dict of str->elem/list/dict (recursively)
            if _is_elem(v):
                return v
            if isinstance(v, Mapping):
                return {
                    str(k): _sanitize_incoming(vv)
                    for k, vv in v.items()
                    if isinstance(k, str)
                }
            if (
                    isinstance(v, Sequence)
                    and not isinstance(v, (str, bytes, bytearray))
            ):
                out = []
                for el in v:
                    se = _sanitize_incoming(el)
                    out.append(se)
                return out
            # anything else is dropped by returning None; caller can still set None
            return None

        filtered = {
            k: _sanitize_incoming(v)
            for k, v in loaded.items()
            if k in allowed
        }

        # construct result with only overrides that differ from defaults (optional, but tidy)
        result = cls()
        for k, v in filtered.items():
            if k in defaults and defaults[k] == v:
                continue
            result[k] = v

        return result

cfg = Cfg._default

if __name__ == '__main__':
    Cfg._default
    Cfg._trace2property
    Cfg._nested
