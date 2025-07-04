"""
Network Initializations
"""

import importlib
from typing import Any, Union

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from tile2net.tiles.cfg.logger import logger
from tile2net.tiles.cfg import cfg


def get_net(criterion: _Loss) -> torch.nn.Module:
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(
        network='tile2net.tiles.tileseg.network.' + cfg.arch,
        num_classes=cfg.DATASET.NUM_CLASSES,
        criterion=criterion
    )
    num_params = sum([param.nelement() for param in net.parameters()])
    logger.debug('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def is_gscnn_arch(cfg: Any) -> bool:
    """
    Network is a GSCNN network
    """
    return 'gscnn' in cfg.arch


def wrap_network_in_dataparallel(net: torch.nn.Module) -> Union[
    DataParallel,
    DistributedDataParallel
]:
    """
    Wrap the network in Dataparallel using PyTorch's native SyncBatchNorm
    """
    if cfg.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=True
        )
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network: str, num_classes: int, criterion: _Loss) -> torch.nn.Module:
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(num_classes=num_classes, criterion=criterion)
    return net
