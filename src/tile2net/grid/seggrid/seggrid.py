from __future__ import annotations

import copy
import gc
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
from functools import *
from pathlib import Path
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
import ctypes
import psutil

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))

if TYPE_CHECKING:
    pass

if False:
    from ..dir import Outdir
    from .filled import Filled
    from .broadcast import Broadcast
    from ..ingrid import InGrid


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _try_malloc_trim() -> None:
    # helper to trim on Linux only
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


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

    @property
    def grid(self):
        return self.instance

    @property
    def ingrid(self) -> InGrid:
        return self.grid

    @property
    def outdir(self) -> Outdir:
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
                f'Predicting segmentation to '
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

            assert wrapper.index.difference(self.index).empty
            assert ingrid.segtile.index.difference(self.index).empty

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
                    # keep a rolling window of just the previous minibatch's futures
                    prev_futures: List = []

                    # note: errors should already exist above; leave as-is
                    # note: iou_acc, pbar, loader, dataset, frame, net, clip, threads are as in your context

                    # for batch in loader:
                    for n, batch in enumerate(loader):
                        # gather batch tensors/indices
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

                        # build the minibatch and submit new work
                        mb = MiniBatch.from_data(
                            images=input_images,
                            net=net,
                            gt_image=labels,
                            grid=grid,
                            threads=threads,
                            clip=clip,
                        )
                        curr_futures = list(mb.submit_all())

                        # update metrics/progress for this batch
                        iou_acc += mb.iou_acc
                        pbar.update(len(input_images))

                        # wait for the previous batch's futures (if any), collect exceptions, then drop them
                        if prev_futures:
                            done, _ = wait(prev_futures)
                            for fut in done:
                                try:
                                    exc = fut.exception()
                                except Exception:
                                    exc = None
                                if exc is not None:
                                    errors.append(exc)
                            prev_futures.clear()

                        # roll the window
                        prev_futures = curr_futures

                        proc = psutil.Process(os.getpid())
                        rss_gb = proc.memory_info().rss / (1024 ** 3)
                        cuda = (
                            torch.cuda.memory_reserved() / (1024 ** 3),
                            torch.cuda.memory_allocated() / (1024 ** 3),
                        )
                        if not n % 100:
                            msg = f'Clearing CPU memory...'
                            logger.debug(msg)
                            try:
                                libc = ctypes.CDLL("libc.so.6")
                                libc.malloc_trim(0)
                            except Exception:
                                pass
                        msg = f'Batch {n} | RSS={rss_gb:.2f} GB | CUDA reserved/alloc={cuda[0]:.2f}/{cuda[1]:.2f} GB'
                        logger.debug(msg)

                    # after loop, make sure the last batch’s futures are done and errors captured
                    if prev_futures:
                        done, _ = wait(prev_futures)
                        for fut in done:
                            try:
                                exc = fut.exception()
                            except Exception:
                                exc = None
                            if exc is not None:
                                errors.append(exc)
                        prev_futures.clear()


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

            # write segmentation benchmark summary to file
            try:
                seg_s = self.sampler.samples
                seg_vals = {
                    'elapsed': seg_s.time_elapsed,
                    'gpu_avg': seg_s.avg_gpu,
                    'gpu_max': seg_s.max_gpu,
                    'vram_avg': seg_s.avg_vram,
                    'vram_max': seg_s.max_vram,
                    'ram_avg': seg_s.avg_ram,
                    'ram_max': seg_s.max_ram,
                    'cpu_avg': seg_s.avg_cpu,
                    'cpu_max': seg_s.max_cpu,
                }

                def _fmt_pct(v: float | None) -> str:
                    return "—" if v is None else f"{v:.1f}%"

                def _fmt_duration(v: float | None) -> str:
                    if v is None:
                        return "—"
                    secs = float(v)
                    if secs < 0:
                        secs = 0.0
                    ms = secs * 1000.0
                    if secs < 1e-3:
                        return "0s"
                    if secs < 1.0:
                        return f"{ms:.0f}ms"
                    days = int(secs // 86400)
                    secs -= days * 86400
                    hours = int(secs // 3600)
                    secs -= hours * 3600
                    minutes = int(secs // 60)
                    secs -= minutes * 60
                    parts = []
                    if days:
                        parts.append(f"{days}d")
                    if hours and len(parts) < 2:
                        parts.append(f"{hours}h")
                    if minutes and len(parts) < 2:
                        parts.append(f"{minutes}m")
                    if len(parts) < 2:
                        parts.append(f"{secs:.1f}s" if secs < 10 else f"{int(secs)}s")
                    return " ".join(parts)

                lines = []
                lines.append("Segmentation benchmark")
                lines.append("======================")
                if seg_vals['elapsed'] is not None:
                    lines.append(f"Time Elapsed: {_fmt_duration(seg_vals['elapsed'])}")
                lines.append(f"GPU Compute: avg {_fmt_pct(seg_vals['gpu_avg'])}, max {_fmt_pct(seg_vals['gpu_max'])}")
                lines.append(f"VRAM Usage: avg {_fmt_pct(seg_vals['vram_avg'])}, max {_fmt_pct(seg_vals['vram_max'])}")
                lines.append(f"RAM Usage:  avg {_fmt_pct(seg_vals['ram_avg'])}, max {_fmt_pct(seg_vals['ram_max'])}")
                lines.append(f"CPU Usage:  avg {_fmt_pct(seg_vals['cpu_avg'])}, max {_fmt_pct(seg_vals['cpu_max'])}")

                summary_path = Path(self.outdir.seggrid.summary)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except Exception as exc:
                logger.warning(f"Could not write segmentation benchmark summary: {exc}")

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
                f'Predicting segmentation to '
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

            assert wrapper.index.difference(self.index).empty
            assert ingrid.segtile.index.difference(self.index).empty

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

            # configure how often to rotate and how many writer threads to use
            # rotate_every: int = cfg.segment.rotate_every if hasattr(cfg.segment, "rotate_every") else 200
            # writer_workers: int = cfg.segment.writer_workers if hasattr(cfg.segment,
            #                                                             "writer_workers") else os.cpu_count() or 4

            rotate_every = 100
            writer_workers = 4

            with ExitStack() as stack, \
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

                # writer threadpool is rotated; start as None
                threads: ThreadPoolExecutor | None = None
                prev_futures: list = []

                # helper to drain/collect exceptions from a set of futures
                def _drain(futs: list) -> None:
                    if not futs:
                        return
                    done, _ = wait(futs)
                    for fut in done:
                        try:
                            exc = fut.exception()
                        except Exception:
                            exc = None
                        if exc is not None:
                            errors.append(exc)
                    futs.clear()

                # helper to rotate the pool (drain -> shutdown -> trim -> recreate)
                def _rotate_pool(n: int) -> None:
                    nonlocal threads, prev_futures
                    # drain any in-flight work from the current pool
                    if threads is not None:
                        logger.debug(
                            f'Rotating writer thread pool at batch {n} '
                            f'(every {rotate_every}); shutting down {writer_workers} workers...'
                        )
                        threads.shutdown(wait=True)
                        threads = None
                        _try_malloc_trim()
                        logger.debug('Writer pool shut down; attempted malloc_trim().')

                    # create a fresh pool and log it
                    threads = ThreadPoolExecutor(max_workers=writer_workers)

                    msg = f'Created new writer thread pool with max_workers={writer_workers} at batch {n}.'
                    logger.debug(msg)

                try:

                    # create the initial pool before the loop
                    _rotate_pool(n=0)

                    for n, batch in enumerate(loader):
                        _drain(prev_futures)

                        # rotate pool at the configured cadence (skip n=0 which already created it)
                        if (
                                n
                                and not n % rotate_every
                        ):
                            _rotate_pool(n=n)

                        # gather batch tensors/indices
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

                        # build the minibatch and submit new work to the *current* pool
                        mb = MiniBatch.from_data(
                            images=input_images,
                            net=net,
                            gt_image=labels,
                            grid=grid,
                            threads=threads,
                            clip=clip,
                        )
                        curr_futures = list(mb.submit_all())

                        # update metrics/progress for this batch
                        iou_acc += mb.iou_acc
                        pbar.update(len(input_images))

                        # roll the window
                        prev_futures = curr_futures

                        # periodic memory logging + optional trim hint
                        proc = psutil.Process(os.getpid())
                        rss_gb = proc.memory_info().rss / (1024 ** 3)
                        cuda = (
                            torch.cuda.memory_reserved() / (1024 ** 3),
                            torch.cuda.memory_allocated() / (1024 ** 3),
                        )

                        msg = (
                            f'Batch {n} | RSS={rss_gb:.2f} GB | '
                            f'CUDA reserved/alloc={cuda[0]:.2f}/{cuda[1]:.2f} GB'
                        )
                        logger.debug(msg)

                    # after loop, make sure the last batch’s futures are done and errors captured
                    _drain(prev_futures)

                finally:
                    # tear down the current pool (if any), then trim
                    try:
                        if threads is not None:
                            logger.debug('Final writer pool shutdown...')
                            threads.shutdown(wait=True)
                            threads = None
                            _try_malloc_trim()
                    except Exception:
                        pass
                    try:
                        pbar.close()
                    except Exception:
                        pass

            # leave your global futures drain as-is if you use it elsewhere
            # wait(futures)
            # for fut in futures:
            #     try:
            #         exc = fut.exception()
            #     except Exception:
            #         exc = None
            #     if exc is not None:
            #         errors.append(exc)
            # futures.clear()

            # write segmentation benchmark summary to file
            try:
                seg_s = self.sampler.samples
                seg_vals = {
                    'elapsed': seg_s.time_elapsed,
                    'gpu_avg': seg_s.avg_gpu,
                    'gpu_max': seg_s.max_gpu,
                    'vram_avg': seg_s.avg_vram,
                    'vram_max': seg_s.max_vram,
                    'ram_avg': seg_s.avg_ram,
                    'ram_max': seg_s.max_ram,
                    'cpu_avg': seg_s.avg_cpu,
                    'cpu_max': seg_s.max_cpu,
                }

                def _fmt_pct(v: float | None) -> str:
                    return "—" if v is None else f"{v:.1f}%"

                def _fmt_duration(v: float | None) -> str:
                    if v is None:
                        return "—"
                    secs = float(v)
                    if secs < 0:
                        secs = 0.0
                    ms = secs * 1000.0
                    if secs < 1e-3:
                        return "0s"
                    if secs < 1.0:
                        return f"{ms:.0f}ms"
                    days = int(secs // 86400)
                    secs -= days * 86400
                    hours = int(secs // 3600)
                    secs -= hours * 3600
                    minutes = int(secs // 60)
                    secs -= minutes * 60
                    parts = []
                    if days:
                        parts.append(f"{days}d")
                    if hours and len(parts) < 2:
                        parts.append(f"{hours}h")
                    if minutes and len(parts) < 2:
                        parts.append(f"{minutes}m")
                    if len(parts) < 2:
                        parts.append(f"{secs:.1f}s" if secs < 10 else f"{int(secs)}s")
                    return " ".join(parts)

                lines = []
                lines.append("Segmentation benchmark")
                lines.append("======================")
                if seg_vals['elapsed'] is not None:
                    lines.append(f"Time Elapsed: {_fmt_duration(seg_vals['elapsed'])}")
                lines.append(f"GPU Compute: avg {_fmt_pct(seg_vals['gpu_avg'])}, max {_fmt_pct(seg_vals['gpu_max'])}")
                lines.append(f"VRAM Usage: avg {_fmt_pct(seg_vals['vram_avg'])}, max {_fmt_pct(seg_vals['vram_max'])}")
                lines.append(f"RAM Usage:  avg {_fmt_pct(seg_vals['ram_avg'])}, max {_fmt_pct(seg_vals['ram_max'])}")
                lines.append(f"CPU Usage:  avg {_fmt_pct(seg_vals['cpu_avg'])}, max {_fmt_pct(seg_vals['cpu_max'])}")

                summary_path = Path(self.outdir.seggrid.summary)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except Exception as exc:
                logger.warning(f"Could not write segmentation benchmark summary: {exc}")

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
