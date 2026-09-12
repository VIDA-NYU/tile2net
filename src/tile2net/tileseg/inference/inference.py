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
from __future__ import absolute_import, annotations, division

import concurrent.futures
import copy
import hashlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import argh
import geopandas as gpd
import numpy
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from geopandas import GeoDataFrame
from PIL import Image
from torch.utils.data import DataLoader

import tile2net.tileseg.network.ocrnet
from tile2net.logger import logger
from tile2net.namespace import Namespace
from tile2net.raster.formats import VectorFormat
from tile2net.raster.pednet import PedNet
from tile2net.raster.project import Project
from tile2net.tileseg import datasets
from tile2net.tileseg import network
from tile2net.tileseg.config import assert_and_infer_cfg, cfg
from tile2net.tileseg.inference.commandline import commandline
from tile2net.tileseg.loss.optimizer import get_optimizer, restore_opt, restore_net
from tile2net.tileseg.loss.utils import get_loss
from tile2net.tileseg.utils.misc import AverageMeter, prep_experiment
from tile2net.tileseg.utils.misc import ThreadedDumper
from tile2net.tileseg.utils.trnval_utils import eval_minibatch

if TYPE_CHECKING:
    from tile2net.raster.raster import Raster
    from tile2net.raster.tile import Tile


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def write_segmentation_png(
        path: str | os.PathLike[str],
        image: np.ndarray,
) -> Path:
    """Atomically persist a lossless, user-viewable segmentation image."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f'.{destination.name}.',
            suffix='.tmp',
            delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
            temporary_path,
            format='PNG',
        )
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def read_segmentation_png(path: str | os.PathLike[str]) -> np.ndarray:
    """Read a persisted RGB segmentation image into memory."""
    with Image.open(path) as image:
        return np.array(image.convert('RGB'), dtype=np.uint8, copy=True)

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
AutoResume = None


_INFERENCE_EVAL_MODES = ('test', 'folder')


def normalize_inference_eval_mode(value: str | None) -> str:
    """Return a supported inference mode with a deterministic default."""
    if value is None:
        return 'test'
    if not isinstance(value, str):
        raise TypeError(
            f'Inference mode must be a string, got {type(value).__name__}.'
        )

    normalized = value.strip().lower()
    if not normalized:
        return 'test'
    if normalized not in _INFERENCE_EVAL_MODES:
        expected = ', '.join(_INFERENCE_EVAL_MODES)
        raise ValueError(
            f'Unsupported inference mode {value!r}; expected one of: {expected}.'
        )
    return normalized


def build_pedestrian_network(
        grid: Raster,
        poly_network: gpd.GeoDataFrame,
        output_format: VectorFormat = VectorFormat.PARQUET,
) -> PedNet | None:
    """Save polygon output and build a network only when features exist."""
    output_format = VectorFormat(output_format)
    polygons = grid.save_ntw_polygons(
        poly_network,
        output_format=output_format,
    )
    if polygons.empty:
        logger.warning(
            'No pedestrian polygons were produced; skipping network creation.'
        )
        return None

    network = PedNet(
        poly=polygons,
        project=grid.project,
        output_format=output_format,
    )
    network.convert_whole_poly2line()
    return network


class Inference:
    Dumper = ThreadedDumper

    def __init__(self, args: Namespace):
        self.args = args
        args.model.eval = normalize_inference_eval_mode(args.model.eval)
        cfg.MODEL.EVAL = args.model.eval
        if not 0 <= args.dump_percent <= 100:
            raise ValueError('dump_percent must be between 0 and 100.')
        if args.dump_percent:
            if not os.path.exists(args.result_dir):
                # os.mkdir(args.result_dir, )
                os.makedirs(args.result_dir, exist_ok=True)
            logger.info(f'Inferencing. Segmentation results will be saved to {args.result_dir}')
        else:
            logger.info('Inferencing. Segmentation results will not be saved.')

        weights = Project.resources.assets.weights
        if (
                args.model.snapshot == weights.satellite_2021.path.absolute().__fspath__()
                and not weights.satellite_2021.path.exists()
        ) or (
                args.model.hrnet_checkpoint == weights.hrnetv2_w48_imagenet_pretrained.path.absolute().__fspath__()
                and not weights.hrnetv2_w48_imagenet_pretrained.path.exists()
        ):
            weights.path.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading weights for segmentation, this may take a while...")
            weights.download()
            logger.info("Weights downloaded.")

        args.best_record = dict(epoch=-1, iter=0, val_loss=1e10, acc=0, acc_cls=0, mean_iu=0, fwavacc=0)

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
        if not torch.cuda.is_available() or num_gpus < 1:
            raise RuntimeError(
                'Tile2Net inference requires a CUDA-capable PyTorch runtime. '
                'Run generation separately on CPU or request a GPU compute node.'
            )
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
            logger.info(
                f'Using CUDA device 0: {torch.cuda.get_device_name(0)}.'
            )

        assert args.result_dir is not None, 'need to define result_dir arg'

    def inference(self, rasterfactory=None):
        train_loader: DataLoader
        val_loader: DataLoader
        train_obj: datasets.Loader
        net: torch.nn.parallel.DataParallel
        optim: torch.optim.sgd.SGD
        args = self.args

        # logx.initialize(
        #     logdir=str(args.result_dir),
        #     tensorboard=True, hparams=vars(args),
        #     global_rank=args.global_rank
        # )
        assert_and_infer_cfg(args)
        prep_experiment(args)
        train_loader, val_loader, train_obj = datasets.setup_loaders(args)
        criterion, criterion_val = get_loss(args)
        if args.model.snapshot:
            if 'ASSETS_PATH' in args.model.snapshot:
                args.model.snapshot = args.model.snapshot.replace('ASSETS_PATH', cfg.ASSETS_PATH)

            # Compute and validate checksum
            expected_checksum = '745f8c099e98f112a152aedba493f61fb6d80c1761e5866f936eb5f361c7ab4d'
            actual_checksum = sha256sum(args.model.snapshot)
            if actual_checksum != expected_checksum:
                raise RuntimeError(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")

            checkpoint = torch.load(
                args.model.snapshot,
                map_location='cpu',
                weights_only=False,
            )
            args.restore_net = True
            msg = "Loading weights from: checkpoint={}".format(args.model.snapshot)
            logger.info(msg)

        net: tile2net.tileseg.network.ocrnet.MscaleOCR = network.get_net(args, criterion)

        optim, scheduler = get_optimizer(args, net)

        net = network.wrap_network_in_dataparallel(args, net)
        if not next(net.parameters()).is_cuda:
            raise RuntimeError('Model parameters were not placed on CUDA.')
        logger.info('CUDA model placement verified.')
        if args.restore_optimizer:
            restore_opt(optim, checkpoint)
        if args.restore_net:
            restore_net(net, checkpoint)
        if args.options.init_decoder:
            net.module.init_mods()
        torch.cuda.empty_cache()

        if args.tile2net:
            if rasterfactory:
                city_data = rasterfactory
            else:
                from tile2net.raster.raster import Raster
                # boundary_path = args.boundary_path
                city_info_path = cfg.CITY_INFO_PATH
                # @maryam boundary path is unused in original
                # boundary_path = cfg.MODEL.boundary_path
                city_data = Raster.from_info(city_info_path)
        else:
            city_data = None

        match args.model.eval:
            case 'test':
                self.validate(
                        val_loader, net, criterion=None, optim=None, epoch=0,
                        calc_metrics=False, dump_assets=args.dump_assets,
                        dump_all_images=True, testing=True, grid=city_data,
                        args=args,
                )
                return 0

            case 'folder':
                # Using a folder for evaluation means to not calculate metrics
                self.validate(
                        val_loader, net, criterion=criterion_val, optim=optim, epoch=0,
                        calc_metrics=False, dump_assets=args.dump_assets,
                        dump_all_images=True,
                        args=args,
                )
                return 0

            case _:
                raise ValueError(
                    f'Unsupported inference mode {args.model.eval!r}.'
                )

    def validate(
            self,
            val_loader: DataLoader,
            net: torch.nn.parallel.DataParallel,
            criterion: tile2net.tileseg.loss.utils.CrossEntropyLoss2d,
            optim: torch.optim.sgd.SGD,
            epoch: int,
            calc_metrics=True,
            dump_assets=False,
            dump_all_images=False,
            testing=None,
            grid: Raster = None,
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
        args = self.args
        # todo map_feature is how Tile generates poly
        gdfs: list[GeoDataFrame] = []

        self.dumper = dumper = self.Dumper(
                val_len=len(val_loader),
                dump_all_images=dump_all_images,
                dump_assets=dump_assets,
                args=args,
        )

        net.eval()
        val_loss = AverageMeter()
        iou_acc = 0
        pred = dict()
        _temp = dict.fromkeys([i for i in range(10)], None)
        for val_idx, data in enumerate(val_loader):
            input_images, labels, img_names, _ = data

            # Run network
            assets, _iou_acc = eval_minibatch(
                    data, net, criterion, val_loss, calc_metrics, args, val_idx,
            )
            iou_acc += _iou_acc
            input_images, labels, img_names, _ = data

            dumpdict = dict(
                    gt_images=labels,
                    input_images=input_images,
                    img_names=img_names,
                    assets=assets,
            )
            if testing:

                dump = dumper.dump(dumpdict, val_idx, testing=True, grid=grid)
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
            if grid:
                # todo: for now we concate from a list of all the polygons generated during the session;
                #   eventually we will serialize all the files and then use dask for batching
                if not gdfs:
                    poly_network = gpd.GeoDataFrame()
                    logging.warning(
                            f'No polygons were dumped'
                    )
                else:
                    poly_network = pd.concat(gdfs)
                del gdfs

                build_pedestrian_network(
                    grid,
                    poly_network,
                    output_format=args.vector_format,
                )



class LocalDumper(ThreadedDumper):
    def map_features(
            self,
            tile: Tile,
            src_img: np.ndarray,
            img_array=True,
            save_segmentation: bool = False,
    ) -> Optional[GeoDataFrame]:
        if save_segmentation:
            future = self.threads.submit(
                write_segmentation_png,
                tile.segmentation,
                src_img,
            )
            self.futures.append(future)
        result = super().map_features(tile, src_img, img_array=img_array)
        for future in self.futures:
            future.result()
        return result


class LocalInference(Inference):
    Dumper = LocalDumper

    def validate(self, *args, grid: Raster, **kwargs):
        if self.args.dump_percent:
            grid.project.segmentation.path.mkdir(exist_ok=True, parents=True)
            for segmentation, tile in zip(
                    grid.project.segmentation.files(),
                    grid.tiles.flat,
            ):
                tile.segmentation = segmentation
        return super().validate(*args, grid=grid, **kwargs)


class RemoteInference(Inference):
    def inference(self, rasterfactory=None):
        from tile2net.raster.raster import Raster
        grid = Raster.from_info(cfg.CITY_INFO_PATH)
        threads = concurrent.futures.ThreadPoolExecutor()
        tile_paths = [
            (tile, path)
            for tile, path in zip(
                grid.tiles.flat,
                grid.project.segmentation.files(),
            )
            if os.path.exists(path)
        ]
        predictions = threads.map(
            read_segmentation_png,
            (path for _, path in tile_paths),
        )
        it_polygons = (
                self.Dumper.map_features(tile, prediction, img_array=True)
                for (tile, _), prediction in zip(tile_paths, predictions)
        )

        gdfs = [
                polygons
                for polygons in it_polygons
                if polygons is not None
        ]
        logger.debug(f'{len(gdfs)} polygons dumped')
        if not len(gdfs):
            logger.error('No polygons were dumped')

        if not gdfs:
            poly_network = gpd.GeoDataFrame()
            logging.warning(
                    f'No polygons were dumped'
            )
        else:
            poly_network = pd.concat(gdfs)
        del gdfs

        network = build_pedestrian_network(
            grid,
            poly_network,
            output_format=self.args.vector_format,
        )
        if network is not None:
            logger.debug(f'{len(network.complete_net)=}')


@commandline
def inference(args: Namespace):
    if args.local:
        inference = LocalInference(args)
    elif args.remote:
        inference = RemoteInference(args)
    else:
        inference = Inference(args)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = inference.inference()
    elapsed = time.perf_counter() - started
    allocated = torch.cuda.max_memory_allocated() / 2 ** 20
    reserved = torch.cuda.max_memory_reserved() / 2 ** 20
    logger.info(
        f'CUDA_METRICS device={torch.cuda.get_device_name(args.local_rank)!r} '
        f'elapsed_seconds={elapsed:.3f} '
        f'peak_allocated_mib={allocated:.1f} '
        f'peak_reserved_mib={reserved:.1f}'
    )
    return result


if __name__ == '__main__':
    """
    --city_info /tmp/tile2net/washington_square_park/tiles/washington_square_park_256_info.json --interactive --dump_percent 10
    
    --city_info /tmp/tile2net/frisco_test_1/tiles/frisco_test_1_256_info.json --interactive --dump_percent 10
    """
    argh.dispatch_command(inference)
