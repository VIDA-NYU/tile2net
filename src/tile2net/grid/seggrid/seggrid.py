from __future__ import annotations
from ...grid.frame.namespace import namespace

import copy
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from functools import *
from typing import *

import PIL.Image
import imageio.v3 as iio
import numpy
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel.data_parallel import DataParallel
from ..tileseg.datasets.satellite import Loader
from tqdm import tqdm
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tile2net.grid.cfg.cfg import assert_and_infer_cfg
from tile2net.grid.cfg.logger import logger
from tile2net.grid.grid.static import Static
from tile2net.grid.tileseg import datasets
from tile2net.grid.tileseg import network
from tile2net.grid.tileseg.loss.optimizer import get_optimizer, restore_opt, restore_net
from tile2net.grid.tileseg.loss.utils import get_loss
from tile2net.grid.tileseg.network.ocrnet import MscaleOCR
from tile2net.grid.tileseg.utils.misc import AverageMeter, prep_experiment
from . import delayed
from .minibatch import MiniBatch
from .vectile import VecTile
from .. import frame
from ..grid import file
from ..grid.grid import Grid
from ..util import recursion_block
from .. import util
from .padded import Padded
from .file import File

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))

if False:
    from .filled import Filled
    from .broadcast import Broadcast
    from ..ingrid import InGrid


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


class SegGrid(
    Grid,
):
    __name__ = 'seggrid'

    def _get(
            self,
            instance: InGrid,
            owner: type[Grid],
    ) -> SegGrid:
        self = namespace._get(self, instance, owner)
        if instance is None:
            return copy.copy(self)
        # cache = instance.__dict__
        # key = self.__name__
        # if key in cache:
        #     result: Self = cache[key]
        #     if result.instance is not instance:
        #         haystack = instance.vectile.index
        #         needles = result.index
        #         loc = needles.isin(haystack)
        #         result = result.loc[loc]
        #
        #         needles = instance.vectile.index
        #         haystack = result.index
        #         loc = needles.isin(haystack)
        #         if not np.all(loc):
        #             msg = 'Not all segmentation tiles implicated by input tiles present'
        #             logger.debug(msg)
        #             del cache[key]
        #             return getattr(instance, self.__name__)

        cache = instance.frame.__dict__
        key = self.__name__
        if key in cache:
            result = cache[key]

        else:
            msg = (
                f'ingrid.{self.__name__} has not been set. You may '
                f'customize the segmentation functionality by using '
                f'`Ingrid.set_segmentation`'
            )
            logger.info(msg)
            cfg = instance.cfg

            scale = cfg.segment.scale
            length = cfg.segment.length
            dimension = cfg.segment.dimension

            if scale:
                instance = instance.set_segmentation(scale=scale)
            elif length:
                instance = instance.set_segmentation(length=length)
            elif dimension:
                instance = instance.set_segmentation(dimension=dimension)
            else:
                raise ValueError(
                    'You must set at least one of the following '
                    'segmentation parameters: segscale, segtile.length, or segdimension.'
                )
            result = instance.seggrid

        result.instance = instance

        return result

    locals().update(
        __get__=_get,
    )

    # @cached_property
    # def length(self) -> int:
    #     """How many input grid comprise a segmentation tile"""
    #     ingrid = self.grid.ingrid
    #     result = 2 ** (ingrid.scale - self.scale)
    #     return result
    #
    # @cached_property
    # def dimension(self):
    #     """How many pixels in a inmentation tile"""
    #     seggrid = self.grid
    #     ingrid = seggrid.ingrid
    #     result = ingrid.dimension * self.length
    #     return result

    @cached_property
    def length(self) -> int:
        """How many input grid comprise a segmentation tile"""
        ingrid = self.grid.ingrid
        result = 2 ** (ingrid.scale - self.scale)
        return result

    @cached_property
    def dimension(self):
        """How many pixels in a inmentation tile"""
        seggrid = self.grid
        ingrid = seggrid.ingrid
        result = ingrid.dimension * self.length
        return result

    # @cached_property
    # def grid(self) -> InGrid:
    #     ...

    @property
    def grid(self):
        return self.instance

    @property
    def ingrid(self) -> InGrid:
        return self.grid

    @property
    def outdir(self):
        return self.ingrid.outdir

    @property
    def cfg(self):
        return self.ingrid.cfg

    @property
    def static(self):
        return self.ingrid.static

    @VecTile
    def vectile(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the grid have been stitched:
        # xtile of the vectile
        self.vecgrid.xtile = ...
        # ytile of vectile
        self.vecgrid.ytile = ...

    @File
    def file(self):
        ...

    @property
    def seggrid(self) -> Self:
        return self

    @property
    def skip(self):
        result = ~self.file.grayscale.apply(os.path.exists)
        return result

    @delayed.Filled
    def filled(self) -> Filled:
        ...

    @delayed.Broadcast
    def broadcast(self) -> Broadcast:
        ...

    @Padded
    def padded(self):
        ...

    @recursion_block
    def predict(
            self,
            force=None,
            batch_size=None,
    ) -> Self:
        if self is not self.filled:
            return self.filled.predict(
                force=force,
                batch_size=batch_size
            )
        grid = self
        # cfg = grid.cfg
        with grid.cfg as cfg:

            # preemptively stitch so logging apears more sequential
            # otherwise you get "now predicting" before "now stitching"
            _ = self.file.infile
            _ = self.padded.infile

            if force is not None:
                cfg.force = force
            if batch_size is not None:
                cfg.model.bs_val = batch_size
            if not cfg.force:
                loc = ~grid.file.grayscale.apply(os.path.exists)
                grid = (
                    grid.frame
                    .loc[loc]
                    .pipe(grid.from_frame, wrapper=grid)
                )
                if not np.any(loc):
                    msg = 'All segmentation grid are on disk.'
                    logger.info(msg)
                    return grid

            if cfg.dump_percent:
                logger.info(f'Inferencing. Segmentation results will be saved to {grid.outdir.seg_results}')
            else:
                logger.info('Inferencing. Segmentation results will not be saved.')

            if (
                    not os.path.exists(cfg.model.snapshot)
                    and cfg.model.snapshot == Static.snapshot
            ) or (
                    not os.path.exists(cfg.model.hrnet_checkpoint)
                    and cfg.model.hrnet_checkpoint == Static.hrnet_checkpoint
            ):
                logger.info('Downloading weights for segmentation, this may take a while...')
                grid.static.download()
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

            struct = datasets.setup_loaders(tiles=grid)
            val_loader = struct.val_loader
            criterion, criterion_val = get_loss(cfg)

            cfg.restore_net = True
            msg = "Loading weights \n\t{}".format(cfg.model.snapshot)
            logger.debug(msg)
            if cfg.model.snapshot != Static.snapshot:
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

            # preemptively stitch resources;
            # nonessential resources are commented out
            _ = (
                grid.file.grayscale,
                # grid.file.infile,
                # grid.file.probability,
                # grid.file.sidebyside,
                # grid.file.segresults,
                # grid.file.error,
                # grid.file.colored,
                # grid.file.submit,
            )

            if cfg.model.eval == 'test':
                self._validate(
                    loader=val_loader,
                    net=net,
                    force=force,
                    grid=grid,
                )

            elif cfg.model.eval == 'folder':
                self._validate(
                    loader=val_loader,
                    net=net,
                    criterion=criterion_val,
                    force=force,
                    grid=grid,
                )

            else:
                raise ValueError(f"Unknown evaluation mode: {cfg.model.eval}. ")

            msg = f'Finished predicting {len(grid)} segmentation tiles.'
            logger.info(msg)

            return self

    def _validate(
            self,
            loader: Loader,
            net: torch.nn.parallel.DataParallel,
            force,
            grid: SegGrid,
            criterion: Optional[tile2net.grid.tileseg.loss.utils.CrossEntropyLoss2d] = None,
    ):
        """
        Run validation for one epoch
        :val_loader: data loader for validation
        """
        GRID = grid
        cfg = self.cfg
        testing = False
        if cfg.model.eval == 'test':
            testing = True

        input_images: torch.Tensor
        labels: torch.Tensor
        img_names: tuple
        prediction: numpy.ndarray
        pred: dict
        values: numpy.ndarray

        net.eval()
        val_loss = AverageMeter()
        iou_acc = 0
        scales = [cfg.default_scale]
        logger.debug(f'Using multi-scale inference (AVGPOOL) with scales {scales}')

        threads = ThreadPoolExecutor()
        futures = []
        batch_size = cfg.model.bs_val
        clip = self.ingrid.dimension * cfg.segment.pad

        msg = f'Inferring to {grid.outdir.seggrid.grayscale.dir}'

        unit = f' {self.seggrid.__name__}.{self.seggrid.file.grayscale.name}'
        with logging_redirect_tqdm(), cfg:
            pbar = tqdm(
                total=len(GRID),  # one tick per tile
                desc=msg,
                unit=unit,
                dynamic_ncols=True,
            )

            for (
                    i,
                    (input_images, labels, img_names, scale_float)
            ) in enumerate(loader):
                start = i * batch_size
                # grid = GRID.iloc[start: start + len(input_images)]
                grid = (
                    GRID.frame
                    .iloc[start: start + len(input_images)]
                    .pipe(GRID.from_frame, wrapper=GRID)
                )

                batch = MiniBatch.from_data(
                    images=input_images,
                    net=net,
                    gt_image=labels,
                    criterion=criterion,
                    val_loss=val_loss,
                    grid=grid,
                    threads=threads,
                    clip=clip
                )
                futures.extend(batch.submit_all())

                iou_acc += batch.iou_acc
                pbar.update(len(input_images))

                if (
                        cfg.options.test_mode
                        and i >= 5
                ):
                    break

            pbar.close()

        wait(futures)
        for fut in futures:
            fut.result()
        futures.clear()

