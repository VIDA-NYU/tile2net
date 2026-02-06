from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import os
import sys
from contextlib import ExitStack
from functools import cached_property
from typing import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel.data_parallel import DataParallel
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tile2net.core.cfg.cfg import Cfg
from tile2net.core.cfg.logger import logger
from tile2net.core.grid.static import Static
from tile2net.core.loaders.sample import SampleDataWrapper, SampleDataLoader, Sample, StreamSample
from tile2net.core.loaders.sampler import DistributedSampler
from tile2net.core.loaders.val import StreamValDataSet, ValDataSet
from tile2net.core.seggrid.gac import grayscale_area_closing
from tile2net.core.seggrid.gmb import geodesic_masked_boosting
from tile2net.core.seggrid.hysteresis import hysteresis_boost
from tile2net.core.seggrid.minibatch import MiniBatch
from tile2net.core.seggrid.submit import Submit
from tile2net.tileseg import network
from tile2net.tileseg.datasets.sampler import DistributedSampler
from tile2net.tileseg.loss.optimizer import get_optimizer, restore_net, restore_opt
from tile2net.tileseg.loss.utils import get_loss
from tile2net.tileseg.network.ocrnet import MscaleOCR
from tile2net.tileseg.utils.misc import AverageMeter


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclasses.dataclass
class Model:
    """
    See also:
        >>> Predict.model
    """
    net: DataParallel
    optim: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    criterion: torch.nn.Module
    criterion_val: torch.nn.Module




class Predict:
    """Semantic segmentation prediction orchestrator."""

    def __init__(
            self,
            cfg: Cfg,
            wrapper: SampleDataWrapper,
            padding: int,
            tile_dimension: int = None,
    ):
        self.cfg = cfg
        self.wrapper = wrapper
        self.padding = padding
        self.tile_dimension = tile_dimension
        self._setup_done = False
        self._stream_stats = {
            'success': 0,
            'empty': 0,
            'not_found': 0,
            'failed': 0,
        }

    def _validate_and_download_checkpoints(self) -> None:
        """Validate and download model checkpoints if needed."""
        # todo: rethink where the weights are to be stored
        cfg = self.cfg
        if (
                not os.path.exists(cfg.model.snapshot)
                and cfg.model.snapshot == Static.snapshot
        ) or (
                not os.path.exists(cfg.model.hrnet_checkpoint)
                and cfg.model.hrnet_checkpoint == Static.hrnet_checkpoint
        ):
            logger.info('Downloading weights for segmentation, this may take a while...')
            Static.download()
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

    def _setup_device_and_distributed(self) -> None:
        """Configure GPU/CPU device and distributed training settings."""
        cfg = self.cfg
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

    def _setup(self) -> None:
        """Perform one-time setup based on the old implementation."""
        if self._setup_done:
            return

        if self.cfg.dump_percent:
            raise NotImplementedError
        else:
            logger.info('Inferencing. Segmentation results will not be saved.')

        self._validate_and_download_checkpoints()
        self._setup_device_and_distributed()

        self._setup_done = True

    @cached_property
    def model(self) -> Model:
        """
        Load and restore the segmentation model (cached).
        The model contains the network, optimizer, scheduler, and loss functions.
        """
        self._setup()
        cfg = self.cfg

        logger.debug('Initializing segmentation model...')
        criterion, criterion_val = get_loss(cfg)

        cfg.restore_net = True
        msg = "Loading weights from \n\t{}".format(cfg.model.snapshot)
        logger.debug(msg)
        if cfg.model.snapshot != Static.snapshot:
            logger.warning(
                f'Weights are being loaded using weights_only=False. '
                f'We assure the security of our weights by using a checksum, '
                f'but you are using a custom path: \n\t{cfg.model.snapshot}. '
            )

        checkpoint = torch.load(
            cfg.model.snapshot,
            map_location='cpu',
            weights_only=False,
        )

        logger.debug('Building network architecture...')
        net: MscaleOCR = network.get_net(criterion)
        optim, scheduler = get_optimizer(net)
        net: DataParallel = network.wrap_network_in_dataparallel(net)

        if cfg.restore_optimizer:
            restore_opt(optim, checkpoint)
        if cfg.restore_net:
            logger.debug('Restoring model weights...')
            restore_net(net, checkpoint)
        if cfg.options.init_decoder:
            net.module.init_mods()

        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

        logger.debug('Model loaded successfully.')
        return Model(
            net=net,
            optim=optim,
            scheduler=scheduler,
            criterion=criterion,
            criterion_val=criterion_val,
        )

    @cached_property
    def dataset(self) -> ValDataSet | StreamValDataSet:
        """
        Instantiate torch Dataset for evaluation/prediction.

        SampleDataWrapper wraps metadata such as input file and position in the
        mosaic for stitching without writing to file:
        >>> SampleDataWrapper.from_columns()

        The ValDataSet returns samples:
        >>> ValDataSet.__getitem__()
        """
        self._setup()
        cfg = self.cfg

        if cfg.model.eval:
            mode = None
            match cfg.model.eval:
                case 'val' | None:
                    mode = 'val'
                case 'trn':
                    mode = 'train'
                case 'folder' | 'test' as mode:
                    pass
                case _:
                    raise ValueError(f"Unknown eval mode: {cfg.model.eval}")

            if cfg.segmentation.stream:
                dataset = StreamValDataSet(
                    wrapper=self.wrapper,
                    tile_dimension=self.tile_dimension,
                    mode=mode,
                    padding=self.padding,
                )
            else:
                dataset = ValDataSet(
                    wrapper=self.wrapper,
                    tile_dimension=self.tile_dimension,
                    mode=mode,
                    padding=self.padding,
                )
        else:
            raise NotImplementedError("Training mode not implemented")

        cfg.dataset_inst = dataset
        return dataset

    @cached_property
    def loader(self):
        """
        Instantiate torch DataLoader for evaluation/prediction.
        """
        cfg = self.cfg
        dataset = self.dataset

        if cfg.distributed:
            sampler = DistributedSampler(
                dataset,
                pad=False,
                permutation=False,
                consecutive_sample=False
            )
        else:
            sampler = None

        # base configuration shared across all modes
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=cfg.validation.batch_size,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True,
        )

        if cfg.segmentation.stream:
            # streaming: high concurrency to mask http latency
            loader = SampleDataLoader(
                **loader_kwargs,
                num_workers=16,
                prefetch_factor=8,
            )
        else:
            # local: moderate concurrency to match cpu core count for decoding
            loader = SampleDataLoader(
                **loader_kwargs,
                num_workers=8,
                prefetch_factor=2,
            )
        return loader

    def __iter__(self) -> Iterator[MiniBatch]:
        """
        Iterates through the metadata encapsulated by the DataWrapper.
        For each dict returned by the DataLoader, it passes the inputs
        through the network via `MiniBatch.from_data`:
            >>> MiniBatch.from_data

        Purpose is to generate MiniBatches which are handled for serialization by these methods:
            >>> Predict.submit_pred
            >>> Predict.submit_prob
        """
        logger.debug('Setting model to evaluation mode...')
        self.model.net.eval()
        val_loss = AverageMeter()
        scales = [self.cfg.default_scale]
        if self.cfg.multi_scale_inference:
            scales.extend(self.cfg.model.extra_scales)
            scales.sort(reverse=True)
            msg = f'Using multi-scale inference (AVGPOOL) with scales {scales}'
        else:
            msg = f'Using single-scale inference with scale {scales}'
        logger.info(msg)

        dataset = self.dataset
        loader = self.loader

        logger.debug(f'Starting inference on {len(dataset)} seg-tiles...')

        unit = ' seg-tiles'
        msg = 'Predicting seg-tiles'

        match self.cfg.segmentation.postprocess:
            case None:
                postprocess = None
                logger.debug('No postprocessing enabled.')
            case 'gac':
                postprocess = grayscale_area_closing
                logger.debug('Using grayscale area closing postprocessing.')
            case 'gmb':
                postprocess = geodesic_masked_boosting
                logger.debug('Using geodesic masked boosting postprocessing.')
            case 'hysteresis':
                postprocess = hysteresis_boost
                logger.debug('Using hysteresis boost postprocessing.')
            case _:
                msg = (
                    f'Unknown segmentation postprocess: '
                    f'{self.cfg.segmentation.postprocess}.'
                    f' Using raw probabilities instead.'
                )
                logger.error(msg)
                postprocess = None

        with ExitStack() as stack, \
                torch.inference_mode(), \
                Submit(workers=self.cfg.compress_workers) as submit:

            stack.enter_context(logging_redirect_tqdm())
            # Register tqdm with the stack; auto-closes on exit/error
            pbar = tqdm(
                total=len(dataset),
                desc=msg,
                unit=unit,
                dynamic_ncols=True,
                mininterval=1,
            )
            pbar = stack.enter_context(pbar)

            for batch in loader:
                batch: Sample | StreamSample
                submit.rotate()

                if self.cfg.segmentation.stream:
                    for key in ('success', 'empty', 'not_found', 'failed'):
                        if key in batch:
                            self._stream_stats[key] += int(batch[key].sum())

                kwargs = dict(
                    images=batch['image'],
                    masks=batch['mask'],
                    net=self.model.net,
                    submit=submit,
                    postprocess=postprocess,
                    padding=self.padding,
                )

                path_keys = [
                    'pred_paths',
                    'prob_paths',
                    'unclipped_prob_paths',
                    'colorized_paths',
                ]

                kwargs.update({
                    key: batch[key]
                    for key in path_keys
                    if key in batch
                })

                mb = MiniBatch.from_data(**kwargs)
                mb.submit_all()

                yield mb
                pbar.update(len(mb))

                if self.cfg.segmentation.stream:
                    pbar.set_postfix(
                        tiles_ok=self._stream_stats['success'],
                        empty=self._stream_stats['empty'],
                        not_found=self._stream_stats['not_found'],
                        failed=self._stream_stats['failed'],
                        refresh=False,
                    )

                # todo: still necessary to set None and save the tensor memory here?
                mb.probs = None
                mb.unclipped_probs = None

        msg = f'Finished predicting {len(dataset)} seg-tiles.'
        logger.debug(msg)

        if self.cfg.segmentation.stream:
            total_tiles = sum(self._stream_stats.values())
            msg = (
                f"Downloaded {self._stream_stats['success']:,} tiles "
                f"({self._stream_stats['empty']} empty, "
                f"{self._stream_stats['not_found']} not found, "
                f"{self._stream_stats['failed']} failed)"
            )
            logger.debug(msg)


def main():
    """
    Parse arguments and run inference.

    Setup:
        >>> Predict._setup()
    Model:
        >>> Predict.model
    Dataset and Loader:
        >>> Predict.dataset
        >>> Predict.loader
    Iterate through minibatches:
        >>> Predict.__iter__()
    """
    parser = argparse.ArgumentParser(description="Semantic segmentation prediction subprocess")
    parser.add_argument("--cfg", type=str, required=True, help="Path to cfg JSON file")
    parser.add_argument("--wrapper", type=str, required=True, help="Path to wrapper Parquet file")
    parser.add_argument("--padding", type=int, help="Padding size for tiles")
    parser.add_argument("--tile-dimension", type=int, help="Dimension of individual tiles")

    args: argparse.Namespace = parser.parse_args()
    cfg: Cfg = Cfg.from_json(args.cfg)
    tile_dimension = args.tile_dimension
    padding = args.padding
    wrapper = SampleDataWrapper.from_parquet(args.wrapper)

    with cfg as cfg:
        predict = Predict(
            cfg=cfg,
            wrapper=wrapper,
            tile_dimension=tile_dimension,
            padding=padding,
        )

        for mb in predict:
            del mb

        del predict, wrapper
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
