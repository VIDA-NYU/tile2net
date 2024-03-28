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
##############################################################################
# Config
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
from tile2net.logger import logger

import torch
from tile2net.logger import logger

from tile2net.tileseg.utils.attr_dict import AttrDict
from tile2net.raster.project import  Project

__C = AttrDict()
cfg = __C
__C.GLOBAL_RANK = 0
__C.EPOCH = 0
# Absolute path to a location to keep some large files.
__C.ASSETS_PATH = None

__C.CITY_INFO_PATH = None

__C.MODEL = AttrDict()
__C.MODEL.SNAPSHOT = None
# Where output results get written
__C.RESULT_DIR = None
__C.DUMP_PERCENT = 0

# Border Relaxation Count
__C.BORDER_WINDOW = 1
# Number of epoch to use before turn off border restriction
__C.REDUCE_BORDER_EPOCH = -1
# Comma Seperated List of class id to relax
__C.STRICTBORDERCLASS = None

__C.OPTIONS = AttrDict()
__C.OPTIONS.TEST_MODE = False
__C.OPTIONS.INIT_DECODER = False
__C.OPTIONS.TORCH_VERSION = None

__C.TRAIN = AttrDict()
__C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10
__C.TRAIN.FP16 = False
__C.NUM_WORKERS = 4
# Attribute Dictionary for Dataset
__C.DATASET = AttrDict()

# __C.DATASET.CENTROID_ROOT = \
#     os.path.join(__C.ASSETS_PATH, 'uniform_centroids')

__C.DATASET.CENTROID_ROOT = None
__C.DATASET.SATELLITE_DIR = None

__C.DATASET.MEAN = [0.485, 0.456, 0.406]
__C.DATASET.STD = [0.229, 0.224, 0.225]
__C.DATASET.NAME = ''
__C.DATASET.NUM_CLASSES = 0
__C.DATASET.IGNORE_LABEL = -1
__C.DATASET.DUMP_IMAGES = False
__C.DATASET.CLASS_UNIFORM_PCT = 0.5
__C.DATASET.CLASS_UNIFORM_TILE = 512
__C.DATASET.COARSE_BOOST_CLASSES = None
__C.DATASET.COLORIZE_MASK_FN = None
__C.DATASET.MASK_OUT_CITYSCAPES = False
__C.DATASET.CV = 0

# This enables there to always be translation augmentation during random crop
# process, even if image is smaller than crop size.
__C.DATASET.TRANSLATE_AUG_FIX = False
__C.DATASET.LANCZOS_SCALES = False
# Use a center crop of size args.model.pre_size for mapillary validation
# Need to use this if you want to dump images
__C.DATASET.MAPILLARY_CROP_VAL = False
# __C.DATASET.CROP_SIZE = '896'
__C.DATASET.CROP_SIZE = '1024,1024'

__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = None
__C.MODEL.MSCALE = False
__C.MODEL.THREE_SCALE = False
__C.MODEL.ALT_TWO_SCALE = False
__C.MODEL.ALIGN_CORNERS = False
__C.MODEL.EXTRA_SCALES = '0.5,1.5'
__C.MODEL.N_SCALES = None
__C.MODEL.MSCALE_LO_SCALE = 0.5
__C.MODEL.OCR_ASPP = False
__C.MODEL.SEGATTN_BOT_CH = 256
__C.MODEL.ASPP_BOT_CH = 256
__C.MODEL.MSCALE_CAT_SCALE_FLT = False
__C.MODEL.MSCALE_INNER_3x3 = True
__C.MODEL.MSCALE_DROPOUT = False
__C.MODEL.MSCALE_OLDARCH = False
__C.MODEL.MSCALE_INIT = 0.5
__C.MODEL.ATTNSCALE_BN_HEAD = False
__C.MODEL.GRAD_CKPT = False


__C.MODEL.WRN38_CHECKPOINT = None
__C.MODEL.WRN41_CHECKPOINT = None
__C.MODEL.X71_CHECKPOINT = None
__C.MODEL.HRNET_CHECKPOINT = None
__C.MODEL.SNAPSHOT = None

__C.LOSS = AttrDict()
# Weight for OCR aux loss
__C.LOSS.OCR_ALPHA = 0.4
# Use RMI for the OCR aux loss
__C.LOSS.OCR_AUX_RMI = False
# Supervise the multi-scale predictions directly
__C.LOSS.SUPERVISED_MSCALE_WT = 0

__C.MODEL.OCR = AttrDict()
__C.MODEL.OCR.MID_CHANNELS = 512
__C.MODEL.OCR.KEY_CHANNELS = 256
__C.MODEL.OCR_EXTRA = AttrDict()
__C.MODEL.OCR_EXTRA.FINAL_CONV_KERNEL = 1
__C.MODEL.OCR_EXTRA.STAGE1 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE1.NUM_MODULES = 1
__C.MODEL.OCR_EXTRA.STAGE1.NUM_RANCHES = 1
__C.MODEL.OCR_EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
__C.MODEL.OCR_EXTRA.STAGE1.NUM_BLOCKS = [4]
__C.MODEL.OCR_EXTRA.STAGE1.NUM_CHANNELS = [64]
__C.MODEL.OCR_EXTRA.STAGE1.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE2 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE2.NUM_MODULES = 1
__C.MODEL.OCR_EXTRA.STAGE2.NUM_BRANCHES = 2
__C.MODEL.OCR_EXTRA.STAGE2.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
__C.MODEL.OCR_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
__C.MODEL.OCR_EXTRA.STAGE2.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE3 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE3.NUM_MODULES = 4
__C.MODEL.OCR_EXTRA.STAGE3.NUM_BRANCHES = 3
__C.MODEL.OCR_EXTRA.STAGE3.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
__C.MODEL.OCR_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
__C.MODEL.OCR_EXTRA.STAGE3.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE4 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE4.NUM_MODULES = 3
__C.MODEL.OCR_EXTRA.STAGE4.NUM_BRANCHES = 4
__C.MODEL.OCR_EXTRA.STAGE4.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
__C.MODEL.OCR_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
__C.MODEL.OCR_EXTRA.STAGE4.FUSE_METHOD = 'SUM'

__C.MODEL.LR_SCHEDULER = 'poly'
__C.MODEL.OPTIMIZER = 'sgd'
__C.MODEL.SCALE_MIN = 0.5
__C.MODEL.SCALE_MAX = 2.0
__C.MODEL.BS_TRN = 2
__C.MODEL.BS_VAL = 1
__C.MODEL.COLOR_AUG = 0.25
__C.MODEL.GBLUR = False
__C.MODEL.BBLUR = True
__C.MODEL.FULL_CROP_MODELING = False
__C.MODEL.EVAL = 'test'
__C.EVAL_FOLDER = None
__C.MODEL.PRE_SIZE = None
__C.MODEL.RAND_AUGMENT = None
__C.MODEL.RMI_LOSS = False
__C.MODEL.IMG_WT_LOSS = True
# everything before here may have been accessed through cfg.ATTR

# these are all not used as cfg.attr but defined for default values across all commandlines
# everything after here is only accessed through args.attr
__C.LR = 0.002
__C.ARCH = 'ocrnet.HRNet_Mscale'
__C.OLD_DATA = False
__C.BATCH_WEIGHTING = True
__C.RESCALE = 1.0
__C.REPOLY = 1.5
__C.LOCAL_RANK = 0
__C.AMSGRAD = None
__C.FREEZE_TRUNK = False
__C.HARDNM = 0
__C.TRUNK = 'hrnetv2'
__C.MAX_EPOCH = 300
__C.MAX_CU_EPOCH = 150
__C.START_EPOCH = 0
__C.BRT_AUG = False
__C.POLY_EXP = 2.0
__C.POLY_STEP = 110
__C.WEIGHT_DECAY = 1e-4
__C.MOMENTUM = 0.9
__C.RESUME = None
__C.RESTORE_OPTIMIZER = False
__C.RESTORE_NET = False
__C.EXP = 'default'
__C.SYNCBN = False
__C.DUMP_AUGMENTATION_IMAGES = False
__C.WT_BOUND = 1.0
__C.SCF = False
__C.MULTI_SCALE_INFERENCE = False
__C.DEFAULT_SCALE = 1.0
__C.LOG_MSINF_TO_TB = False
__C.DO_FLIP = False
__C.AMP_OPT_LEVEL = 'O1'
__C.DUMP_TOPN = 50
__C.DUMP_ASSETS = False
__C.DUMP_ALL_IMAGES = False
__C.DUMP_TOPN_ALL = True
__C.ONLY_COARSE = False
__C.TRIAL = None
__C.VAL_FREQ = 1
__C.DETERMINISTIC = False
__C.SUMMARY = False
__C.CALC_METRICS = True
__C.OCR_AUX_LOSS_RMI = False
__C.TAU_FACTOR = 1
__C.TILE2NET = True
__C.BOUNDARY_PATH = None
__C.NGPU = torch.cuda.device_count()
__C.WORLD_SIZE = 1
__C.DISTRIBUTED = False

__C.INTERACTIVE = False

def torch_version_float():
    version_str = torch.__version__
    version_re = re.search(r'^([0-9]+\.[0-9]+)', version_str)
    if version_re:
        version = float(version_re.group(1))
        # logger.debug(f'Torch version: {version}, {version_str}')
    else:
        version = 1.0
        # logger.debug(f'Can\'t parse torch version ({version}), assuming {version}')
    return version


def assert_and_infer_cfg(args, train_mode=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg
    settings during script execution (which can lead to hard to debug errors
    or code that's harder to understand than is necessary).
    """

    __C.OPTIONS.TORCH_VERSION = torch_version_float()
    __C.MODEL.BNFUNC = torch.nn.BatchNorm2d

    if not train_mode:
        cfg.immutable(True)
        return

    cfg.DATASET.NAME = args.dataset.name
    cfg.DATASET.DUMP_IMAGES = args.dump_augmentation_images

    cfg.DATASET.CLASS_UNIFORM_BIAS = None


    cfg.MODEL.MSCALE = ('mscale' in args.arch.lower() or 'attnscale' in
                        args.arch.lower())

    def str2list(s):
        alist = s.split(',')
        alist = [float(x) for x in alist]
        return alist

    if args.model.n_scales:
        cfg.MODEL.N_SCALES = str2list(args.model.n_scales)
        logger.debug('n scales {}'.format(cfg.MODEL.N_SCALES))

    __C.RESULT_DIR = args.result_dir

    __C.DATASET.CROP_SIZE = '1024,1024'


#      # todo fixme: make all code use this cfg

def update_epoch(epoch):
    # Update EPOCH CTR
    cfg.immutable(False)
    cfg.EPOCH = epoch
    cfg.immutable(True)


def update_dataset_cfg(num_classes, ignore_label):
    cfg.immutable(False)
    cfg.DATASET.NUM_CLASSES = num_classes
    cfg.DATASET.IGNORE_LABEL = ignore_label
    # logger.debug('num_classes = {}'.format(num_classes))
    logger.debug('num_classes = {}'.format(num_classes))
    cfg.immutable(True)


def update_dataset_inst(dataset_inst):
    cfg.immutable(False)
    cfg.DATASET_INST = dataset_inst
    cfg.immutable(True)
