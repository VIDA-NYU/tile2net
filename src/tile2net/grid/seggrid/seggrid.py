from __future__ import annotations

import copy
import gc
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from contextlib import ExitStack
from functools import *
from typing import *
from typing import TYPE_CHECKING

import numpy
import torch
import torch.distributed as dist
from torch.nn.parallel.data_parallel import DataParallel
from tqdm import tqdm
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tile2net.grid.cfg.cfg import assert_and_infer_cfg
from tile2net.grid.cfg.logger import logger
from tile2net.grid.grid.static import Static
from tile2net.tileseg.loss.optimizer import get_optimizer, restore_opt, restore_net
from tile2net.tileseg.loss.utils import get_loss
from tile2net.tileseg.network.ocrnet import MscaleOCR
from tile2net.tileseg.utils.misc import AverageMeter, prep_experiment
from . import delayed
from .file import File
from .minibatch import MiniBatch
from .padded import Padded
from .vectile import VecTile
from ..grid.grid import Grid
from ..loaders.sample import SampleDataWrapper
from ..loaders.val import ValDataSet, ValDataLoader
from ..util import recursion_block
from ...grid.frame.namespace import namespace
from ...tileseg import datasets, network

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))

if TYPE_CHECKING:
    pass

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

            scale = cfg.segmentation.scale
            length = cfg.segmentation.length
            dimension = cfg.segmentation.dimension

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
    ) -> None:
        if self is not self.filled:
            return self.filled.predict(
            )
            self.file.grayscale.map(os.path.exists)
        grid = self

        errors = []
        outer_exc = None
        with grid.cfg as cfg:
            if cfg.dump_percent:
                raise NotImplementedError
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

            criterion, criterion_val = get_loss(cfg)

            cfg.restore_net = True
            msg = "Loading weights from \n\t{}".format(cfg.model.snapshot)
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

            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()

            if cfg.model.eval == 'folder':
                criterion = criterion_val
            elif cfg.model.eval == 'test':
                criterion = None
            else:
                raise ValueError(f"Unknown evaluation mode: {cfg.model.eval}. ")

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
            if cfg.multi_scale_inference:
                scales.extend(cfg.model.extra_scales.split(','))
                msg = f'Using multi-scale inference (AVGPOOL) with scales {scales}'
            else:
                msg = f'Using single-scale inference with scale {scales}'
            logger.debug(msg)
            clip = self.ingrid.dimension * cfg.segmentation.pad

            msg = (
                f'Using stitched imagery from '
                f'\n\t{self.outdir.seggrid.infile.dir} '
                f'\nand predicting segmentation to '
                f'\n\t{self.outdir.seggrid.grayscale.dir}'
            )
            logger.info(msg)

            ingrid = self.ingrid.broadcast
            force = ~ingrid.segtile.grayscale.map(os.path.exists)
            force |= self.cfg.force
            wrapper: SampleDataWrapper = SampleDataWrapper.from_tiles(
                infile=ingrid.file.infile,
                mask=[None] * len(ingrid),
                index=ingrid.segtile.index,
                background=0,
                row=ingrid.segtile.r,
                col=ingrid.segtile.c,
                force=force,
            )
            if wrapper.empty:
                msg = f'All segmentation tiles are already on disk.'
                logger.info(msg)
                return
            dataset = ValDataSet.from_wrapper(wrapper)
            loader: ValDataLoader = dataset.loader

            GRID = self.filled
            frame = GRID
            assert wrapper.index.difference(frame.index).empty

            expected: Self = (
                self.broadcast
                .loc[~self.broadcast.index.duplicated()]
                .loc[wrapper.index.unique()]
            )
            expected.file.grayscale.map(os.path.exists)

            frame = self.loc[~self.index.duplicated()]

            unit = f' {self.__class__.__name__}.{self.file.grayscale.name}'
            msg = 'Predicting Segmentation Tiles'
            futures = []

            t = time.time()
            with ExitStack() as stack, \
                    ThreadPoolExecutor() as threads, \
                    torch.inference_mode(), \
                    self.sampler:
                stack.enter_context(logging_redirect_tqdm())
                stack.enter_context(cfg)
                pbar = tqdm(
                    total=len(dataset),
                    desc=msg,
                    unit=unit,
                    dynamic_ncols=True,
                )
                try:
                    for batch in loader:
                        input_images = batch['input']
                        labels = batch['mask']
                        i = (
                            batch['i']
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        loc = dataset.index[i]
                        grid = frame.loc[loc].copy()

                        mb = MiniBatch.from_data(
                            images=input_images,
                            net=net,
                            gt_image=labels,
                            grid=grid,
                            threads=threads,
                            clip=clip,
                        )
                        futures.extend(mb.submit_all())

                        iou_acc += mb.iou_acc
                        pbar.update(len(input_images))

                        # todo: reintroduce
                        # if cfg.options.test_mode and (i >= 5 if hasattr(i, "__len__") is False else i[0] >= 5):
                        #     break
                finally:
                    try:
                        pbar.close()
                    except Exception:
                        pass

                # wait for all submitted tasks and collect exceptions
                wait(futures)
                for fut in futures:
                    try:
                        exc = fut.exception()
                    except Exception:
                        exc = None
                    if exc is not None:
                        errors.append(exc)
                futures.clear()

            t = time.time() - t
            msg = (
                f'Adding {t:.1f}s to total '
                f'{self.instance.__class__.__name__} time usage.'
            )
            logger.debug(msg)
            self.instance.time_usage += t

            if errors:
                try:
                    raise ExceptionGroup("worker task errors", errors)  # Py 3.11+
                except NameError:
                    primary, *rest = errors
                    for ex in rest:
                        if not hasattr(primary, "__notes__"):
                            try:
                                primary.add_note(str(ex))  # Py 3.11 add_note if available
                            except Exception:
                                pass
                    raise primary

            msg = f'Finished predicting {len(dataset)} segmentation tiles.'
            logger.info(msg)

            for name in (
                'dataset loader wrapper input_images labels net '
                'batch optim scheduler criterion criterion_val struct'
            ).split():
                if name in locals():
                    del locals()[name]
            gc.collect()
            torch.cuda.empty_cache()

        assert ingrid.segtile.grayscale.map(os.path.exists).all()

    @cached_property
    def disk_usage(self) -> int:
        result = self.broadcast.file.disk_usage.sum()
        return result

    @cached_property
    def time_usage(self):
        return 0
