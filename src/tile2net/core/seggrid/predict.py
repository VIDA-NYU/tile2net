from __future__ import annotations
from typing import *

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
import torchvision.transforms as standard_transforms
from torch.nn.parallel.data_parallel import DataParallel
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import tile2net.tileseg.transforms.transforms as extended_transforms
from tile2net.core.basegrid.static import Static
from tile2net.core.cfg.cfg import Cfg
from tile2net.core.cfg.logger import logger
from tile2net.core.loaders.sample import SampleDataWrapper, SampleDataLoader, Sample
from tile2net.core.loaders.sampler import DistributedSampler
from tile2net.core.loaders.val import ValDataSet, ValDataLoader
from tile2net.core.seggrid.minibatch import MiniBatch
from tile2net.core.seggrid.submit import Submit
from tile2net.tileseg import network
from tile2net.tileseg.datasets.sampler import DistributedSampler
from tile2net.tileseg.loss.optimizer import get_optimizer, restore_net, restore_opt
from tile2net.tileseg.loss.utils import get_loss
from tile2net.tileseg.network.ocrnet import MscaleOCR
from tile2net.tileseg.utils.misc import AverageMeter

if TYPE_CHECKING:
    from tile2net.core.loaders.image import ImageDataSet
    from tile2net.core.loaders.mask import MaskDataSet


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


@dataclasses.dataclass
class Data:
    wrapper: SampleDataWrapper
    loader: ValDataLoader
    set: ValDataSet

    @classmethod
    def from_wrapper(
            cls,
            wrapper: SampleDataWrapper,
            cfg: Cfg,
    ):
        """
        Instantiate the torch DataLoader and DataSet using the DataWrapper class,
        encapsulating these into a dataclass.
        """
        # todo: model.eval is a str, not bool
        if cfg.model.eval:
            out = cls._from_eval(wrapper=wrapper, cfg=cfg)
        else:
            out = cls._from_train(wrapper=wrapper, cfg=cfg)

        cfg.dataset_inst = out.set
        return out

    @classmethod
    def _from_train(
            cls,
            wrapper: SampleDataWrapper,
            cfg: Cfg,
    ):
        """Instantiate Data for training."""
        raise NotImplementedError

        mean = cfg.dataset.mean
        std = cfg.dataset.std
        train_input_transform = []
        color_aug = cfg.model.color_aug
        if color_aug:
            item = extended_transforms.ColorJitter(
                brightness=color_aug,
                contrast=color_aug,
                saturation=color_aug,
                hue=color_aug
            )
            train_input_transform.append(item)
        if cfg.model.bblur:
            item = extended_transforms.RandomBilateralBlur()
            train_input_transform.append(item)
        elif cfg.model.gblur:
            item = extended_transforms.RandomGaussianBlur()
            train_input_transform.append(item)
        train_input_transform.extend((
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean, std)
        ))
        train_input_transform = standard_transforms.Compose(train_input_transform)

        target_train_transform = standard_transforms.Compose([])

    @classmethod
    def _from_eval(
            cls,
            wrapper: SampleDataWrapper,
            cfg: Cfg,
    ):
        """Instantiate Data for evaluation/prediction."""

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

        bs_val = cfg.validation.batch_size or 1
        dataset = ValDataSet(wrapper=wrapper, mode=mode)
        if cfg.distributed:
            sampler = DistributedSampler(
                dataset,
                pad=False,
                permutation=False,
                consecutive_sample=False
            )
        else:
            sampler = None
        loader = SampleDataLoader(
            dataset=dataset,
            batch_size=bs_val,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
        )
        out = cls(
            wrapper=wrapper,
            loader=loader,
            set=dataset,
        )
        return out


class Predict:
    """Semantic segmentation prediction orchestrator."""

    def __init__(
            self,
            cfg: Cfg,
            wrapper: SampleDataWrapper,
            clip: int,
    ):
        self.cfg = cfg
        self.wrapper = wrapper
        self.clip = clip
        self._setup_done = False

    def _validate_and_download_checkpoints(self) -> None:
        """Validate and download model checkpoints if needed."""
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

        return Model(
            net=net,
            optim=optim,
            scheduler=scheduler,
            criterion=criterion,
            criterion_val=criterion_val,
        )

    @cached_property
    def data(self) -> Data:
        """
        Instantiate dataclass, containing DataWrapper, torch Dataset, and torch DataLoader.

        During eval:
            >>> Data._from_eval()
        During train:
            >>> Data._from_train()

        SampleDataWrapper wraps metadata such as input file and position in the
        mosaic for stitching without writing to file:
        >>> SampleDataWrapper.from_columns()

        The ValDataSet returns samples:
        >>> ValDataSet.__getitem__()

        The ValDataLoader isn't interesting. The __iter__ type hints are just overridden:
        >>> ValDataLoader.__iter__()
        """
        self._setup()
        return Data.from_wrapper(self.wrapper, self.cfg)

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
        self.model.net.eval()
        val_loss = AverageMeter()
        scales = [self.cfg.default_scale]
        if self.cfg.multi_scale_inference:
            scales.extend(self.cfg.model.extra_scales)
            scales.sort(reverse=True)
            msg = f'Using multi-scale inference (AVGPOOL) with scales {scales}'
        else:
            msg = f'Using single-scale inference with scale {scales}'
        logger.debug(msg)

        logger.info('Predicting segmentation masks')

        # Dataset and DataLoader are constructed from the DataWrapper
        dataset = self.data.set
        loader: ValDataLoader = self.data.loader

        unit = ' seg-tiles'
        msg = 'Predicting seg-tiles'

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
                batch: Sample
                submit.rotate()
                # instantiate MB from the dict; this will run the forward pass
                kwargs = dict(
                    images=batch['image'],
                    masks=batch['mask'],
                    net=self.model.net,
                    pred_paths=batch['pred_paths'],
                    prob_paths=batch['prob_paths'],
                    submit=submit,
                    clip=self.clip,
                )
                if 'unclipped_prob_paths' in batch:
                    kwargs['unclipped_prob_paths'] = batch['unclipped_prob_paths']

                mb = MiniBatch.from_data(**kwargs)

                yield mb
                pbar.update(len(mb))
                # todo: still necessary to set None and save the tensor memory here?
                mb.probs = None
                mb.unclipped_probs = None

        msg = f'Finished predicting {len(dataset)} seg-tiles.'
        logger.info(msg)

    def submit_pred(self) -> Iterator[MiniBatch]:
        """
        Iterate through minibatches and submit only predictions.
        This is useful when probability scores are not needed, such as simply generating networks for end-users.
        """
        for mb in self:
            mb.submit_pred()
            yield mb

    def submit_prob(self) -> Iterator[MiniBatch]:
        """
        Iterate through minibatches and submit both predictions and probabilities.
        This is useful when probability scores are needed, such as testing and visualization.
        """
        for mb in self:
            mb.submit_pred()
            mb.submit_prob()
            yield mb

    def submit_all(self) -> Iterator[MiniBatch]:
        """
        Iterate through minibatches and submit predictions, probabilities, and unclipped probabilities.
        This is useful when unclipped probabilities are needed for postprocessing.
        """
        for mb in self:
            mb.submit_pred()
            mb.submit_prob()
            mb.submit_unclipped_prob()
            yield mb


def main():
    """
    Parse arguments and run inference.

    Setup:
        >>> Predict._setup()
    Model:
        >>> Predict.model
    Data:
        >>> Predict.data
    Iterate through minibatches:
        >>> Predict.__iter__()
    Submit predictions/probabilities:
        >>> Predict.submit_pred()
        >>> Predict.submit_prob()
        >>> Predict.submit_all()
    """
    parser = argparse.ArgumentParser(description="Semantic segmentation prediction subprocess")
    parser.add_argument("--cfg", type=str, required=True, help="Path to cfg JSON file")
    parser.add_argument("--wrapper", type=str, required=True, help="Path to wrapper Parquet file")
    parser.add_argument("--clip", type=int, required=True, help="Clipping value for padding")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        choices=['unclipped_prob', 'prob', 'pred'],
        help="Output mode: unclipped_prob (all 3), prob (prob+pred), or pred (pred only)"
    )

    args: argparse.Namespace = parser.parse_args()

    cfg: Cfg = Cfg.from_json(args.cfg)
    clip = args.clip
    wrapper = SampleDataWrapper.from_parquet(args.wrapper)
    output = args.output

    with cfg as cfg:
        predict = Predict(
            cfg=cfg,
            wrapper=wrapper,
            clip=clip,
        )

        match output:
            case 'unclipped_prob':
                for mb in predict.submit_all():
                    del mb
            case 'prob':
                for mb in predict.submit_prob():
                    del mb
            case 'pred':
                for mb in predict.submit_pred():
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
