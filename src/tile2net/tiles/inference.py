from __future__ import annotations

import datetime
import os
import shutil
import warnings

from tile2net.raster.tile_utils.topology import fill_holes, replace_convexhull

os.environ['USE_PYGEOS'] = '0'
from tile2net.raster.tile_utils.geodata_utils import (
    buff_dfs,
)

warnings.simplefilter(action='ignore', category=FutureWarning)
from tile2net.raster.tile_utils.genutils import (
    createfolder,
)

import geopandas as gpd
import logging
import numpy
import os
import pandas as pd
import sys
import torch
import torch.distributed as dist
from geopandas import GeoDataFrame
from torch.utils.data import DataLoader
from typing import Optional

import tile2net.tiles.tileseg.network.ocrnet
from tile2net.logger import logger
from tile2net.raster.pednet import PedNet
from tile2net.tiles.tileseg import datasets
from tile2net.tiles.tileseg import network
from tile2net.tiles.tileseg.config import assert_and_infer_cfg
from tile2net.tiles.tileseg.loss.optimizer import get_optimizer, restore_opt, restore_net
from tile2net.tiles.tileseg.loss.utils import get_loss
from tile2net.tiles.tileseg.utils.misc import AverageMeter, prep_experiment
from tile2net.tiles.tileseg.utils.misc import ThreadedDumper
from tile2net.tiles.tileseg.utils.trnval_utils import eval_minibatch
from tile2net.tiles.cfg import cfg
from tile2net.tiles.tileseg.utils.misc import DumpDict
args = cfg

if False:
    pass
import hashlib


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
AutoResume = None

if False:
    from .tiles import Tiles


class Inference(

):
    Dumper = ThreadedDumper

    @property
    def best_record(self):
        return dict(
            epoch=-1,
            iter=0,
            val_loss=1e10,
            acc=0,
            acc_cls=0,
            mean_iu=0,
            fwavacc=0
        )

    def __init__(
            self,
            tiles: Tiles,
            crs=3857,
    ):
        self.tiles = tiles
        self.crs = crs

        if args.dump_percent:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f'Inferencing. Segmentation results will be saved to {tiles.outdir.seg_results}')
        else:
            logger.info('Inferencing. Segmentation results will not be saved.')

        # if (
        #         args.model.snapshot == Tiles.static.snapshot
        #         and not os.path.exists(args.model.snapshot)
        # ) or (
        #         args.model.hrnet_checkpoint == Tiles.static.hrnet_checkpoint
        #         and not os.path.exists(args.model.hrnet_checkpoint)
        # ):
        args.model.snapshot
        args.model.hrnet_checkpoint
        args.model.snapshot
        args.model.hrnet_checkpoint
        if (
            not os.path.exists(args.model.snapshot)
            or not os.path.exists(args.model.hrnet_checkpoint)
        ):
            logger.info('Downloading weights for segmentation, this may take a while...')
            tiles.static.download()
            if not os.path.exists(args.model.snapshot):
                msg = (
                    f'Downloaded snapshot not found: {args.model.snapshot}. '
                    f'You must have passed a custom path that does not exist.'
                )
                raise FileNotFoundError(msg)
            if not os.path.exists(args.model.hrnet_checkpoint):
                msg = (
                    f'Downloaded HRNet checkpoint not found: {args.model.hrnet_checkpoint}. '
                    f'You must have passed a custom path that does not exist.'
                )
                raise FileNotFoundError(msg)
            expected_checksum = '745f8c099e98f112a152aedba493f61fb6d80c1761e5866f936eb5f361c7ab4d'
            actual_checksum = sha256sum(args.model.snapshot)
            if actual_checksum != expected_checksum:
                raise RuntimeError(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")

        args.best_record = self.best_record

        # Enable CUDNN Benchmarking optimization
        torch.backends.cudnn.benchmark = True
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        args.world_size = 1

        # Test Mode run two epochs with a few iterations of training and val
        if args.options.test_mode:
            args.max_epoch = 2

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            if args.model.eval == 'test':
                # Single GPU setup
                logger.info('Using a single GPU.')
                args.local_rank = 0
                torch.cuda.set_device(args.local_rank)
            else:
                # Distributed training setup
                if "RANK" not in os.environ:
                    raise ValueError("You need to launch the process with torch.distributed.launch to \
                    set RANK environment variable")
                args.world_size = int(os.environ.get('WORLD_SIZE', num_gpus))
                dist.init_process_group(backend='nccl', init_method='env://')
                args.local_rank = dist.get_rank()
                torch.cuda.set_device(args.local_rank)
                args.distributed = True
                args.global_rank = int(os.environ['RANK'])
                logger.info(f'Using distributed training with {args.world_size} GPUs.')
        elif num_gpus == 1:
            # Single GPU setup
            args.local_rank = 0
            torch.cuda.set_device(args.local_rank)
            logger.info('Using a single GPU.')
        else:
            # CPU setup
            logger.info('Using CPU. This is not recommended for inference.')
            args.local_rank = -1  # Indicating CPU usage

        # raise NotImplementedError('check this')
        # assert args.result_dir is not None, 'need to define result_dir arg'

        assert_and_infer_cfg(args)
        prep_experiment(args)
        train_loader, val_loader, train_obj = datasets.setup_loaders()
        criterion, criterion_val = get_loss(args)

        args.restore_net = True
        msg = "Loading weights from: checkpoint={}".format(args.model.snapshot)
        logger.info(msg)
        if (
                args.model.snapshot != Tiles.static.snapshot
        ):
            msg = (
                f'Weights are being loaded using weights_only=False. '
                f'We assure the security of our weights by using a checksum, '
                f'but you are using a custom path: {args.model.snapshot}. '
            )
            logger.warning(msg)

        checkpoint = torch.load(
            args.model.snapshot,
            map_location='cpu',
            weights_only=False,
        )

        net: tile2net.tiles.tileseg.network.ocrnet.MscaleOCR = network.get_net(args, criterion)
        optim, scheduler = get_optimizer(args, net)

        net = network.wrap_network_in_dataparallel(args, net)
        if args.restore_optimizer:
            restore_opt(optim, checkpoint)
        if args.restore_net:
            restore_net(net, checkpoint)
        if args.options.init_decoder:
            net.module.init_mods()
        torch.cuda.empty_cache()

        if args.model.eval == 'test':
            self.validate(
                val_loader=val_loader,
                net=checkpoint['net'],
                criterion=None,
                optim=None,
                epoch=0,
                calc_metrics=False,
                dump_assets=args.dump_assets,
                dump_all_images=True,
                testing=True,
                args=args,
            )

        elif args.model.eval == 'folder':
            self.validate(
                val_loader=val_loader,
                net=checkpoint['net'],
                criterion=criterion_val,
                optim=optim,
                epoch=0,
                calc_metrics=False,
                dump_assets=args.dump_assets,
                dump_all_images=True,
                testing=False,
                args=args,
            )

        else:
            raise ValueError(f'Unknown eval option: {args.model.eval}')

    def validate(
            self,
            val_loader: DataLoader,
            net: torch.nn.parallel.DataParallel,
            criterion: Optional[tile2net.tiles.tileseg.loss.utils.CrossEntropyLoss2d] = None,
            optim: Optional[torch.optim.Optimizer] = None,
            epoch: int = 0,
            calc_metrics=False,
            dump_assets=False,
            dump_all_images=True,
            testing=False,
            **kwargs
    ):
        """
        Run validation for one epoch
        :val_loader: data loader for validation
        """
        input_images: torch.Tensor
        labels: torch.Tensor
        img_names: tuple
        prediction: numpy.ndarray
        pred: dict
        values: numpy.ndarray
        args = cfg
        gdfs: list[GeoDataFrame] = []

        # todo map_feature is how Tile generates poly
        self.dumper = dumper = self.Dumper(
            val_len=len(val_loader),
            dump_all_images=dump_all_images,
            dump_assets=dump_assets,
            args=args,
        )

        tiles = self.tiles
        net.eval()
        val_loss = AverageMeter()
        iou_acc = 0
        pred = dict()
        _temp = dict.fromkeys([i for i in range(10)], None)
        for val_idx, data in enumerate(val_loader):
            input_images, labels, img_names, _ = data

            # Run network
            assets, _iou_acc = eval_minibatch(
                data=data,
                net=net,
                criterion=criterion,
                val_loss=val_loss,
                calc_metrics=calc_metrics,
                args=args,
                val_idx=val_idx,
            )
            iou_acc += _iou_acc
            input_images, labels, img_names, _ = data

            dumpdict = DumpDict(
                gt_images=labels,
                input_images=input_images,
                img_names=img_names,
                assets=assets,
            )
            if testing:
                dump = dumper.dump(dumpdict, val_idx, testing=True, tiles=tiles)
            else:
                dump = dumper.dump(dumpdict, val_idx)
            gdfs.extend(dump)

            if (
                    args.options.test_mode
                    and val_idx > 5
            ):
                break

            if val_idx % 20 == 0:
                logger.debug(f'Inference [Iter: {val_idx + 1} / {len(val_loader)}]')

        if testing:
            # todo: for now we concate from a list of all the polygons generated during the session;
            #   eventually we will serialize all the files and then use dask for batching
            if not gdfs:
                poly_network = gpd.GeoDataFrame()
                logging.warning(f'No polygons were dumped')
            else:
                poly_network = pd.concat(gdfs)
            del gdfs

            self.save_ntw_polygons(poly_network)
            polys = self.ntw_poly
            net = PedNet(poly=polys)
            net.convert_whole_poly2line()

    def save_ntw_polygons(
            self,
            poly_network: gpd.GeoDataFrame,
            crs_metric: int = 3857,
    ):
        """
        Collects the polygons of all tiles created in the segmentation process
        and saves them as a shapefile

        Parameters
        ----------
        poly_network : gpd.GeoDataFrame
            The concatenated GeoDataFrame formed from the polygons of each tile.
        crs_metric : int
            The desired coordinate reference system to save the network polygon with.
        """
        # poly_fold = self.project.polygons.path
        poly_fold = self
        createfolder(poly_fold)
        poly_network.reset_index(drop=True, inplace=True)
        poly_network.set_crs(self.crs, inplace=True)
        if poly_network.crs != crs_metric:
            poly_network.to_crs(crs_metric, inplace=True)
        poly_network.geometry = poly_network.simplify(0.6)
        unioned = buff_dfs(poly_network)
        unioned.geometry = unioned.geometry.simplify(0.9)
        unioned = unioned[unioned.geometry.notna()]
        unioned['geometry'] = unioned.apply(fill_holes, args=(25,), axis=1)
        simplified = replace_convexhull(unioned)
        simplified = simplified[simplified.geometry.notna()]
        simplified = simplified[['geometry', 'f_type']]
        simplified.to_crs(self.crs, inplace=True)

        self.ntw_poly = simplified
        path = os.path.join(
            poly_fold,
            f'{self.name}-Polygons-{datetime.datetime.now().strftime("%d-%m-%Y_%H_%M")}'
        )
        if os.path.exists(path):
            shutil.rmtree(path)
        simplified.to_file(path)
        logging.info('Polygons are generated and saved!')
