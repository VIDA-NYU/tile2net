"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Dataset setup and loaders
"""

import importlib
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

import tile2net.tileseg.transforms.joint_transforms as joint_transforms
import tile2net.tileseg.transforms.transforms as extended_transforms
from runx.logx import logx

from tile2net.tileseg.config import cfg, update_dataset_cfg, update_dataset_inst
from tile2net.tileseg.datasets.randaugment import RandAugment
from toolz import pipe, curried

def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    # TODO add error checking to make sure class exists
    #logx.msg(f'dataset = {args.dataset.dataset.name}')
    dataset = 'satellite'
    # mod = importlib.import_module('tile2net.tileseg.datasets.{}'.format(args.dataset.dataset.name))
    mod = importlib.import_module('tile2net.tileseg.datasets.{}'.format(dataset))
    dataset_cls = getattr(mod, 'Loader')

    logx.msg(f'ignore_label = {dataset_cls.ignore_label}')

    update_dataset_cfg(num_classes=dataset_cls.num_classes,
                       ignore_label=dataset_cls.ignore_label)

    ######################################################################
    # Define transformations, augmentations
    ######################################################################

    # Joint transformations that must happen on both image and mask
    # crop_size = cfg.MODEL.CROPSIZE
    # if ',' in cfg.MODEL.CROP_SIZE:
    #     crop_size = [int(x) for x in crop_size.split(',')]
    # else:
    #     crop_size = int(crop_size)
    crop_size = cfg.DATASET.CROP_SIZE
    if ',' in crop_size:
        crop_size = pipe(
            crop_size.split(','),
            curried.map(int),
            list
        )
    else:
        crop_size = int(crop_size)

    train_joint_transform_list = [
        # TODO FIXME: move these hparams into cfg
        joint_transforms.RandomSizeAndCrop(crop_size,
                                           False,
                                           scale_min=cfg.MODEL.SCALE_MIN,
                                           scale_max=cfg.MODEL.SCALE_MAX,
                                           full_size=cfg.MODEL.FULL_CROP_MODELING,
                                           pre_size=cfg.MODEL.PRE_SIZE)]
    train_joint_transform_list.append(
        joint_transforms.RandomHorizontallyFlip())

    if cfg.MODEL.RAND_AUGMENT is not None:
        N, M = [int(i) for i in cfg.MODEL.RAND_AUGMENT.split(',')]
        assert isinstance(N, int) and isinstance(M, int), \
            f'Either N {N} or M {M} not integer'
        train_joint_transform_list.append(RandAugment(N, M))

    ######################################################################
    # Image only augmentations
    ######################################################################
    train_input_transform = []

    if cfg.MODEL.COLOR_AUG:
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=cfg.MODEL.COLOR_AUG,
            contrast=cfg.MODEL.COLOR_AUG,
            saturation=cfg.MODEL.COLOR_AUG,
            hue=cfg.MODEL.COLOR_AUG)]
    if cfg.MODEL.BBLUR:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif cfg.MODEL.GBLUR:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]

    mean_std = (cfg.DATASET.MEAN, cfg.DATASET.STD)
    train_input_transform += [standard_transforms.ToTensor(),
                              standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()

    target_train_transform = extended_transforms.MaskToTensor()

    if cfg.MODEL.EVAL == 'folder':
        val_joint_transform_list = None
    elif 'mapillary' in args.dataset.name.lower():
        if cfg.MODEL.PRE_SIZE is None:
            eval_size = 2177
        else:
            eval_size = cfg.MODEL.PRE_SIZE
        if cfg.DATASET.MAPILLARY_CROP_VAL:
            val_joint_transform_list = [
                joint_transforms.ResizeHeight(eval_size),
                joint_transforms.CenterCropPad(eval_size)]
        else:
            val_joint_transform_list = [
                joint_transforms.Scale(eval_size)]
    else:
        val_joint_transform_list = None

    if cfg.MODEL.EVAL is None or cfg.MODEL.EVAL == 'val':
        val_name = 'val'
    elif cfg.MODEL.EVAL == 'trn':
        val_name = 'train'
    elif cfg.MODEL.EVAL == 'folder':
        val_name = 'folder'
    elif cfg.MODEL.EVAL == 'test':
        val_name = 'test'

    else:
        raise 'unknown eval mode {}'.format(cfg.MODEL.EVAL)

    ######################################################################
    # Create loaders
    ######################################################################
    val_set = dataset_cls(
        mode=val_name,
        joint_transform_list=val_joint_transform_list,
        img_transform=val_input_transform,
        label_transform=target_transform,
        eval_folder=cfg.EVAL_FOLDER)
        # eval_folder=cfg.MODEL.EVAL_FOLDER)

    update_dataset_inst(dataset_inst=val_set)

    if cfg.MODEL.APEX:
        from tile2net.tileseg.datasets.sampler import DistributedSampler
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False,
                                         consecutive_sample=False)
    else:
        val_sampler = None

    val_loader = DataLoader(val_set, batch_size=cfg.MODEL.BS_VAL,
                            num_workers=cfg.NUM_WORKERS // 2,
                            shuffle=False, drop_last=False,
                            sampler=val_sampler)

    if cfg.MODEL.EVAL is not None:
        # Don't create train dataloader if eval
        train_set = None
        train_loader = None
    else:
        train_set = dataset_cls(
            mode='train',
            joint_transform_list=train_joint_transform_list,
            img_transform=train_input_transform,
            label_transform=target_train_transform)

        if cfg.MODEL.APEX:
            from tile2net.tileseg.datasets.sampler import DistributedSampler
            train_sampler = DistributedSampler(train_set, pad=True,
                                               permutation=True,
                                               consecutive_sample=False)
            train_batch_size = cfg.MODEL.BS_TRN
        else:
            train_sampler = None
            train_batch_size = cfg.MODEL.BS_TRN * args.ngpu

        train_loader = DataLoader(train_set, batch_size=train_batch_size,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=(train_sampler is None),
                                  drop_last=True, sampler=train_sampler)

    return train_loader, val_loader, train_set
