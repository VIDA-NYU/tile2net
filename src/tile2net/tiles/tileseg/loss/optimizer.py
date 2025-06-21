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
"""

# Optimizer and scheduler related tasks

import math
import torch

from torch import optim
from tile2net.tiles.cfg.logger import logger

from tile2net.tiles.cfg import cfg
from tile2net.tiles.tileseg.loss.radam import RAdam


def get_optimizer(net):
    """
    Decide Optimizer (Adam or SGD)
    """
    param_groups = net.parameters()

    if cfg.MODEL.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(param_groups,
                              lr=cfg.lr,
                              weight_decay=cfg.weight_decay,
                              momentum=cfg.momentum,
                              nesterov=False)
    elif cfg.MODEL.OPTIMIZER == 'adam':
        optimizer = optim.Adam(param_groups,
                               lr=cfg.lr,
                               weight_decay=cfg.weight_decay,
                               amsgrad=cfg.amsgrad)
    elif cfg.MODEL.OPTIMIZER == 'radam':
        optimizer = RAdam(param_groups,
                          lr=cfg.lr,
                          weight_decay=cfg.weight_decay)
    else:
        raise ValueError('Not a valid optimizer')

    def poly_schd(epoch):
        return math.pow(1 - epoch / cfg.max_epoch, cfg.poly_exp)

    def poly2_schd(epoch):
        if epoch < cfg.poly_step:
            poly_exp = cfg.poly_exp
        else:
            poly_exp = 2 * cfg.poly_exp
        return math.pow(1 - epoch / cfg.max_epoch, poly_exp)

    if cfg.MODEL.LR_SCHEDULER == 'scl-poly':
        if cfg.REDUCE_BORDER_EPOCH == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_EPOCH
        scale_value = cfg.rescale
        lambda1 = lambda epoch: \
             math.pow(1 - epoch / cfg.MODEL.MAX_EPOCH,
                      cfg.poly_exp) if epoch < rescale_thresh else scale_value * math.pow(
                          1 - (epoch - rescale_thresh) / (cfg.max_epoch - rescale_thresh),
                          cfg.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif cfg.MODEL.LR_SCHEDULER == 'poly2':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=poly2_schd)
    elif cfg.MODEL.LR_SCHEDULER == 'poly':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=poly_schd)
    else:
        raise ValueError('unknown lr schedule {}'.format(cfg.model.lr_scheduler))

    return optimizer, scheduler


def load_weights(net, optimizer, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    # logger.debug("Loading weights from model {}".format(snapshot_file))
    logger.info("Loading weights from model {}".format(snapshot_file))
    net, optimizer = restore_snapshot(net, optimizer, snapshot_file, restore_optimizer_bool)
    return net, optimizer


def restore_snapshot(net, optimizer, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logger.debug("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer


def restore_opt(optimizer, checkpoint):
    assert 'optimizer' in checkpoint, 'cant find optimizer in checkpoint'
    optimizer.load_state_dict(checkpoint['optimizer'])


def restore_net(net, checkpoint):
    assert 'state_dict' in checkpoint, 'cant find state_dict in checkpoint'
    forgiving_state_restore(net, checkpoint['state_dict'])


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """

    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        new_k = k
        if new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size():
            new_loaded_dict[k] = loaded_dict[new_k]
        else:            
            logger.debug("Skipped loading parameter {}".format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net
