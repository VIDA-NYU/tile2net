"""
Network Initializations
"""

import importlib
import torch

from tile2net.logger import logger
from tile2net.tiles.tileseg.config import cfg
from tile2net.logger import logger


def get_net(args, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(network='tile2net.tiles.tileseg.network.' + args.arch,
                    num_classes=cfg.DATASET.NUM_CLASSES,
                    criterion=criterion)
    num_params = sum([param.nelement() for param in net.parameters()])
    # logger.debug('Model params = {:2.1f}M'.format(num_params / 1000000))
    logger.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def is_gscnn_arch(args):
    """
    Network is a GSCNN network
    """
    return 'gscnn' in args.arch


def wrap_network_in_dataparallel(args, net):
    """
    Wrap the network in Dataparallel using PyTorch's native SyncBatchNorm
    """
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    if args.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network, num_classes, criterion):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(num_classes=num_classes, criterion=criterion)
    return net
