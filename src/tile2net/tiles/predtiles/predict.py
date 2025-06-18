from __future__ import annotations, absolute_import, division
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tqdm.auto import tqdm
import geopandas as gpd

import os
import warnings

from torch.nn.parallel.data_parallel import DataParallel

from tile2net.tiles.tileseg.network.ocrnet import MscaleOCR
from .mask2poly import Mask2Poly
from .tileseg.datasets.satellite import labels

os.environ['USE_PYGEOS'] = '0'

warnings.simplefilter(action='ignore', category=FutureWarning)

from tile2net.tiles.static import static
import logging
import numpy
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import tile2net.tiles.tileseg.network.ocrnet
from tile2net.logger import logger
from tile2net.tiles.tileseg import datasets
from tile2net.tiles.tileseg import network
from tile2net.tiles.cfg.cfg import assert_and_infer_cfg
from tile2net.tiles.tileseg.loss.optimizer import get_optimizer, restore_opt, restore_net
from tile2net.tiles.tileseg.loss.utils import get_loss
from tile2net.tiles.tileseg.utils.misc import AverageMeter, prep_experiment
from .minibatch import MiniBatch
from tile2net.tiles.pednet import PedNet

import os
import sys

from typing import *
from typing import Union

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

def __get__(
        self: Predict,
        instance: Tiles,
        owner: type[Tiles],
) -> Predict:
    self.tiles = instance
    self.Tiles = owner
    return self


class Predict:
    tiles: Tiles

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...

    def with_polygons(
            self,
            max_hole_area: Union[
                float,
                dict[str, float],
            ] = None,
            grid_size: int = None,
            min_polygon_area: Union[
                float,
                dict[str, float],
            ] = None,
            convexity: Union[
                float,
                dict[str, float]
            ] = None,
            simplify: Union[
                float,
                dict[str, float]
            ] = None,
    ) -> Self:
        cfg = self.tiles.cfg
        if max_hole_area is not None:
            cfg.polygon.max_hole_area = max_hole_area
        if grid_size is not None:
            cfg.polygon.grid_size = grid_size
        if min_polygon_area is not None:
            cfg.polygon.min_polygon_area = min_polygon_area
        if convexity is not None:
            cfg.polygon.convexity = convexity
        if simplify is not None:
            cfg.polygon.simplify = simplify
        return self

    def to_outdir(
            self,
            force=None,
            batch_size: int = None
    ):
        tiles = self.tiles
        cfg = tiles.cfg

        if force is not None:
            cfg.force = force
        if batch_size is not None:
            cfg.model.bs_val = batch_size

        with cfg:
            if cfg.dump_percent:
                logger.info(f'Inferencing. Segmentation results will be saved to {tiles.outdir.seg_results}')
            else:
                logger.info('Inferencing. Segmentation results will not be saved.')

            if (
                    not os.path.exists(cfg.model.snapshot)
                    and cfg.model.snapshot == static.snapshot
            ) or (
                    not os.path.exists(cfg.model.hrnet_checkpoint)
                    and cfg.model.hrnet_checkpoint == static.hrnet_checkpoint
            ):
                logger.info('Downloading weights for segmentation, this may take a while...')
                tiles.static.download()
                logger.info('Weights downloaded successfully.')
                expected_checksum = '745f8c099e98f112a152aedba493f61fb6d80c1761e5866f936eb5f361c7ab4d'
                actual_checksum = sha256sum(cfg.model.snapshot)
                if actual_checksum != expected_checksum:
                    raise RuntimeError(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")

            if not os.path.exists(cfg.model.hrnet_checkpoint):
                msg = f'HRNet checkpoint not found: {cfg.model.hrnet_checkpoint}. ' \
                      f'You must have passed a custom path that does not exist.'
                raise FileNotFoundError(msg)
            if not os.path.exists(cfg.model.snapshot):
                msg = f'Snapshot not found: {cfg.model.snapshot}. ' \
                      f'You must have passed a custom path that does not exist.'
                raise FileNotFoundError(msg)

            # Enable CUDNN Benchmarking optimization
            torch.backends.cudnn.benchmark = True
            if cfg.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            cfg.world_size = 1

            # Test Mode run two epochs with a few iterations of training and val
            if cfg.options.test_mode:
                cfg.max_epoch = 2

            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                if cfg.model.eval == 'test':
                    # Single GPU setup
                    logger.info('Using a single GPU.')
                    cfg.local_rank = 0
                    torch.cuda.set_device(cfg.local_rank)
                else:
                    # Distributed training setup
                    if "RANK" not in os.environ:
                        raise ValueError("You need to launch the process with torch.distributed.launch to \
                        set RANK environment variable")
                    cfg.world_size = int(os.environ.get('WORLD_SIZE', num_gpus))
                    dist.init_process_group(backend='nccl', init_method='env://')
                    cfg.local_rank = dist.get_rank()
                    torch.cuda.set_device(cfg.local_rank)
                    cfg.distributed = True
                    cfg.global_rank = int(os.environ['RANK'])
                    logger.info(f'Using distributed training with {cfg.world_size} GPUs.')
            elif num_gpus == 1:
                # Single GPU setup
                cfg.local_rank = 0
                torch.cuda.set_device(cfg.local_rank)
                logger.info('Using a single GPU.')
            else:
                # CPU setup
                logger.info('Using CPU. This is not recommended for inference.')
                cfg.local_rank = -1  # Indicating CPU usage

            assert_and_infer_cfg(cfg)
            prep_experiment()
            struct = datasets.setup_loaders(tiles=tiles)
            val_loader = struct.val_loader
            criterion, criterion_val = get_loss(cfg)

            cfg.restore_net = True
            msg = "Loading weights \n\t{}".format(cfg.model.snapshot)
            logger.info(msg)
            if cfg.model.snapshot != static.snapshot:
                msg = (
                    f'Weights are being loaded using weights_only=False. '
                    f'We assure the security of our weights by using a checksum, '
                    f'but you are using a custom path: \n\t{cfg.model.snapshot}. '
                )
                logger.warning(msg)

            checkpoint = torch.load(
                cfg.model.snapshot,
                map_location='cpu',
                weights_only=False,
            )

            net: MscaleOCR = network.get_net(criterion)
            optim, scheduler = get_optimizer(net)

            net: DataParallel = network.wrap_network_in_dataparallel(net)
            if cfg.restore_optimizer:
                restore_opt(optim, checkpoint)
            if cfg.restore_net:
                restore_net(net, checkpoint)
            if cfg.options.init_decoder:
                net.module.init_mods()
            torch.cuda.empty_cache()

            if cfg.model.eval == 'test':
                self._validate(
                    loader=val_loader,
                    net=net,
                    force=force,
                )
            elif cfg.model.eval == 'folder':
                self._validate(
                    loader=val_loader,
                    net=net,
                    criterion=criterion_val,
                    force=force,
                )
            else:
                raise ValueError(f"Unknown evaluation mode: {cfg.model.eval}. ")

    def _validate(
            self,
            loader: DataLoader,
            net: torch.nn.parallel.DataParallel,
            force,
            criterion: Optional[tile2net.tiles.tileseg.loss.utils.CrossEntropyLoss2d] = None,
    ):
        """
        Run validation for one epoch
        :val_loader: data loader for validation
        """
        cfg = self.tiles.cfg
        testing = False
        if cfg.model.eval == 'test':
            testing = True

        input_images: torch.Tensor
        labels: torch.Tensor
        img_names: tuple
        prediction: numpy.ndarray
        pred: dict
        values: numpy.ndarray

        tiles = self.tiles
        net.eval()
        val_loss = AverageMeter()
        iou_acc = 0
        scales = [cfg.default_scale]
        logger.debug(f'Using multi-scale inference (AVGPOOL) with scales {scales}')

        it = enumerate(loader)
        for i, (input_images, labels, img_names, scale_float) in it:
            if i % 20 == 0:
                logger.debug(f'Inference [Iter: {i + 1} / {len(loader)}]')

        with logging_redirect_tqdm():
            for i, (input_images, labels, img_names, scale_float) in enumerate(tqdm(loader)):
                if i % 20 == 0:
                    logger.debug(f'Inference [Iter: {i + 1} / {len(loader)}]')

            batch = (
                MiniBatch
                .from_data(
                    images=input_images,
                    net=net,
                    gt_image=labels,
                    criterion=criterion,
                    val_loss=val_loss,
                    tiles=tiles,
                )
                .submit_all()
                .await_all()
            )
            iou_acc += batch.iou_acc
            if (
                    cfg.options.test_mode
                    and i >= 5
            ):
                break

        else:
            if testing:
                file = tiles.outdir.polygons.file
                if os.path.exists(file):
                    logger.debug(f'Loading existing polygons: \n\t{file}')
                    net = PedNet.from_parquet(
                        file,
                        checkpoint='./checkpoint'
                    )
                else:
                    msg = f'Polygons file not found: {file}. '
                    logger.debug(msg)
                    msg = f'Postprocessing segmentation polygons'
                    logger.info(msg)
                    polys = (
                        tiles.outdir.polygons.files()
                        .pipe(Mask2Poly.from_parquets)
                        .postprocess()
                    )
                    msg = (
                        f'Done. Writing polygons to '
                        f'\n\t{tiles.outdir.polygons.file}'
                    )
                    logger.info(msg)
                    _ = (
                        polys
                        .to_crs(4326)
                        .to_parquet(tiles.outdir.polygons.file)
                    )

                    msg = f'Polygons file not written! {file}'
                    assert os.path.exists(file), msg
                    if polys.empty:
                        logging.warning('No polygons were generated during the session.')
                    net = PedNet.from_polygons(
                        polys,
                        checkpoint='./checkpoint'
                    )

                msg = f'Generating network from polygons'
                logger.info(msg)
                clipped = net.center.clipped

                msg = f'Writing network to\n\t{tiles.outdir.network.file}'
                logger.info(msg)
                _ = (
                    clipped
                    .to_crs(4326)
                    .to_parquet(tiles.outdir.network.file)
                )
