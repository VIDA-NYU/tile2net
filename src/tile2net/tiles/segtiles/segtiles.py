from __future__ import annotations
from ..util import assert_perfect_overlap

import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pathlib import Path
from typing import *

import PIL.Image
import PIL.Image
import imageio.v2
import imageio.v3
import imageio.v3
import imageio.v3 as iio
import numpy
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import tile2net.tiles.tileseg.network.ocrnet
from tile2net.tiles.cfg.cfg import assert_and_infer_cfg
from tile2net.tiles.cfg.logger import logger
from tile2net.tiles.dir.loader import Loader
from tile2net.tiles.tiles.static import Static
from tile2net.tiles.tileseg import datasets
from tile2net.tiles.tileseg import network
from tile2net.tiles.tileseg.loss.optimizer import get_optimizer, restore_opt, restore_net
from tile2net.tiles.tileseg.loss.utils import get_loss
from tile2net.tiles.tileseg.network.ocrnet import MscaleOCR
from tile2net.tiles.tileseg.utils.misc import AverageMeter, prep_experiment
from .minibatch import MiniBatch
from .vectile import VecTile
from ..tiles import file
from ..tiles import tile
from ..tiles.tiles import Tiles
from ...tiles.util import recursion_block
# from .padded import Padded
from . import delayed

if False:
    from .padded import Padded
    from .broadcast import Broadcast
    from ..intiles import InTiles


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
AutoResume = None


class Tile(
    tile.Tile
):
    tiles: SegTiles

    @tile.cached_property
    def length(self) -> int:
        """How many input tiles comprise a segmentation tile"""
        intiles = self.tiles.intiles
        result = 2 ** (intiles.tile.scale - self.scale)
        return result

    @tile.cached_property
    def dimension(self):
        """How many pixels in a inmentation tile"""
        segtiles = self.tiles
        intiles = segtiles.intiles
        result = intiles.tile.dimension * self.length
        return result


def __get__(
        self: SegTiles,
        instance: Tiles,
        owner: type[Tiles],
) -> SegTiles:
    if instance is None:
        return self
    try:
        result = instance.attrs[self.__name__]
    except KeyError as e:
        msg = (
            f'InTiles must be segtiles using `InTiles.stitch` for '
            f'example `InTiles.stitch.to_resolution(2048)` or '
            f'`InTiles.stitch.to_cluster(16)`'
        )
        raise ValueError(msg) from e
    result.intiles = instance
    result.instance = instance
    return result


class File(
    file.File
):
    tiles: SegTiles

    @property
    def stitched(self) -> pd.Series:
        """
        A file for each segmentation tile: the stitched input tiles.
        Stitches input files when segtiles.file is accessed
        """
        tiles = self.tiles
        key = 'file.stitched'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.segtiles.files(tiles)
        tiles[key] = files
        if (
                not tiles.stitch
                and not files.map(os.path.exists).all()
        ):
            tiles.stitch()
        tiles[key] = files
        return tiles[key]

    @property
    def prediction(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.prediction'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.prediction.files(tiles)
        if (
                not tiles.predict
                and not files.map(os.path.exists).all()
        ):
            tiles.predict()
        tiles[key] = files
        return tiles[key]

    @property
    def probability(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.probability'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.seg_results.prob.files(tiles)
        if (
                not tiles.predict
                and not files.map(os.path.exists).all()
        ):
            tiles.predict()
        tiles[key] = files
        return tiles[key]

    @property
    def error(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.error'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.seg_results.error.files(tiles)
        if (
                not tiles.predict
                and not files.map(os.path.exists).all()
        ):
            tiles.predict()
        tiles[key] = files
        return tiles[key]

    @property
    def sidebyside(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.sidebyside'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.seg_results.sidebyside.files(tiles)
        if (
                not tiles.predict
                and not files.map(os.path.exists).all()
        ):
            tiles.predict()
        tiles[key] = files
        return tiles[key]

    @property
    def segresults(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.segresults'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.seg_results.files(tiles)
        if (
                not tiles.predict
                and not files.map(os.path.exists).all()
        ):
            tiles.predict()
        tiles[key] = files
        return tiles[key]

    @property
    def submit(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.submit'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.submit.files(tiles)
        if (
                not tiles.predict
                and not files.map(os.path.exists).all()
        ):
            tiles.predict()
        tiles[key] = files
        return tiles[key]

    @property
    def mask(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.mask'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.mask.files(tiles)
        if (
                not tiles.predict
                and not files.map(os.path.exists).all()
        ):
            tiles.predict()
        tiles[key] = files
        return tiles[key]

    @property
    def maskraw(self) -> pd.Series:
        tiles = self.tiles
        key = 'file.maskraw'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.raw.files(tiles)
        if (
                not tiles.predict
                and not files.map(os.path.exists).all()
        ):
            tiles.predict()
        tiles[key] = files
        return tiles[key]

    def output(self, dirname: str) -> pd.Series:
        tiles = self.tiles
        key = f'file.output.{dirname}'
        if key in tiles:
            return tiles[key]
        files = tiles.intiles.outdir.outputs.files(tiles, dirname)
        if (
                not tiles.predict
                and not files.map(os.path.exists).all()
        ):
            tiles.predict()
        tiles[key] = files
        return tiles[key]


class SegTiles(
    Tiles,
):
    __name__ = 'segtiles'
    locals().update(
        __get__=__get__,
    )

    @tile.cached_property
    def intiles(self) -> InTiles:
        ...

    @recursion_block
    def stitch(self) -> Self:
        if self is not self.padded:
            return self.padded.stitch()

        intiles = self.intiles.padded
        segtiles = self.padded

        loc = ~intiles.segtile.stitched.map(os.path.exists)
        infiles = intiles.file.infile.loc[loc]
        row = intiles.segtile.r.loc[loc]
        col = intiles.segtile.c.loc[loc]
        group = intiles.segtile.stitched.loc[loc]

        intiles.segtile.stitched.map(os.path.exists).all()
        segtiles.padded.file.stitched.map(os.path.exists).all()

        loc = ~segtiles.file.stitched.map(os.path.exists)
        stitched = segtiles.file.stitched.loc[loc]
        n_missing = np.sum(loc)
        n_total = len(segtiles)
        if n_missing == 0:  # nothing to do
            msg = f'All {n_total:,} mosaics are already stitched.'
            logger.info(msg)
            return intiles
        else:
            logger.info(f'Stitching {n_missing:,} of {n_total:,} mosaics missing on disk.')

        loader = Loader(
            files=infiles,
            row=row,
            col=col,
            tile_shape=intiles.tile.shape,
            mosaic_shape=segtiles.tile.shape,
            group=group
        )

        seen = set()
        for f in stitched:
            d = Path(f).parent
            if d not in seen:  # avoids extra mkdir syscalls
                d.mkdir(parents=True, exist_ok=True)
                seen.add(d)

        executor = ThreadPoolExecutor()
        imwrite = imageio.v3.imwrite
        it = loader
        it = tqdm(it, 'stitching', n_missing, unit=' mosaic')

        writes = []
        for path, array in it:
            future = executor.submit(imwrite, path, array)
            writes.append(future)
        for w in writes:
            w.result()

        executor.shutdown(wait=True)
        assert segtiles.file.stitched.map(os.path.exists).all()
        return self

    @tile.cached_property
    def intiles(self) -> InTiles:
        """InTiles object that this SegTiles object is based on"""

    @property
    def ipred(self) -> pd.Series:
        key = 'ipred'
        if key in self.columns:
            return self[key]
        self[key] = np.arange(len(self), dtype=np.uint32)
        return self[key]

    @property
    def outdir(self):
        return self.intiles.outdir

    @property
    def cfg(self):
        return self.intiles.cfg

    @property
    def static(self):
        return self.intiles.static

    @VecTile
    def vectile(self):
        # This code block is just semantic sugar and does not run.
        # These columns are available once the tiles have been stitched:
        # xtile of the vectile
        self.vectiles.xtile = ...
        # ytile of vectile
        self.vectiles.ytile = ...

    @Tile
    def tile(self):
        ...

    @File
    def file(self):
        ...

    @property
    def segtiles(self) -> Self:
        return self

    def preview(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
    ) -> PIL.Image.Image:

        files: pd.Series = self.file.stitched
        R: pd.Series = self.r  # 0-based row id
        C: pd.Series = self.c  # 0-based col id

        dim = self.tile.dimension  # original tile side length
        n_rows = int(R.max()) + 1
        n_cols = int(C.max()) + 1
        div_px = 1 if divider else 0

        # full mosaic size before optional down-scaling
        full_w0 = n_cols * dim + div_px * (n_cols - 1)
        full_h0 = n_rows * dim + div_px * (n_rows - 1)

        scale = 1.0
        if max(full_w0, full_h0) > maxdim:
            scale = maxdim / max(full_w0, full_h0)

        tile_w = max(1, int(round(dim * scale)))
        tile_h = tile_w  # square tiles
        full_w = n_cols * tile_w + div_px * (n_cols - 1)
        full_h = n_rows * tile_h + div_px * (n_rows - 1)

        canvas_col = divider if divider else (0, 0, 0)
        mosaic = Image.new('RGB', (full_w, full_h), color=canvas_col)

        def load(idx: int) -> tuple[int, int, np.ndarray]:
            arr = iio.imread(files.iat[idx])
            if scale != 1.0:
                arr = np.asarray(
                    Image.fromarray(arr).resize(
                        (tile_w, tile_h), Image.Resampling.LANCZOS
                    )
                )
            return R.iat[idx], C.iat[idx], arr

        with ThreadPoolExecutor() as pool:
            for r, c, arr in pool.map(load, range(len(files))):
                x0 = c * (tile_w + div_px)
                y0 = r * (tile_h + div_px)
                mosaic.paste(Image.fromarray(arr), (x0, y0))

        return mosaic

    def view(
            self,
            maxdim: int = 2048,
            divider: Optional[str] = None,
            file: str = 'mask'
    ) -> PIL.Image.Image:

        files = getattr(self.file, file)
        R: pd.Series = self.r  # 0-based row id
        C: pd.Series = self.c  # 0-based col id
        self.intiles.tile.dimension

        dim = self.tile.dimension  # original tile side length
        n_rows = int(R.max()) + 1
        n_cols = int(C.max()) + 1
        div_px = 1 if divider else 0

        # full mosaic size before optional down-scaling
        full_w0 = n_cols * dim + div_px * (n_cols - 1)
        full_h0 = n_rows * dim + div_px * (n_rows - 1)

        scale = 1.0
        if max(full_w0, full_h0) > maxdim:
            scale = maxdim / max(full_w0, full_h0)

        tile_w = max(1, int(round(dim * scale)))
        tile_h = tile_w  # square tiles
        full_w = n_cols * tile_w + div_px * (n_cols - 1)
        full_h = n_rows * tile_h + div_px * (n_rows - 1)

        canvas_col = divider if divider else (0, 0, 0)
        mosaic = Image.new('RGB', (full_w, full_h), color=canvas_col)

        def load(idx: int) -> tuple[int, int, np.ndarray]:
            arr = iio.imread(files.iat[idx])
            if scale != 1.0:
                arr = np.asarray(
                    Image.fromarray(arr).resize(
                        (tile_w, tile_h), Image.Resampling.LANCZOS
                    )
                )
            return R.iat[idx], C.iat[idx], arr

        with ThreadPoolExecutor() as pool:
            for r, c, arr in pool.map(load, range(len(files))):
                x0 = c * (tile_w + div_px)
                y0 = r * (tile_h + div_px)
                mosaic.paste(Image.fromarray(arr), (x0, y0))

        return mosaic

    @property
    def skip(self):
        result = ~self.file.prediction.apply(os.path.exists)
        return result

    @delayed.Padded
    def padded(self) -> Padded:
        ...

    @delayed.Broadcast
    def broadcast(self) -> Broadcast:
        ...

    @recursion_block
    def predict(
            self,
            force=None,
            batch_size=None,
    ) -> Self:
        if self is not self.padded:
            return self.padded.predict(
                force=force,
                batch_size=batch_size
            )
        tiles = self
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
                    and cfg.model.snapshot == Static.snapshot
            ) or (
                    not os.path.exists(cfg.model.hrnet_checkpoint)
                    and cfg.model.hrnet_checkpoint == Static.hrnet_checkpoint
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

            if not cfg.force:
                loc = ~tiles.file.prediction.apply(os.path.exists)
                tiles: SegTiles = tiles.loc[loc].copy()

            _ = (
                tiles.file.prediction,
                tiles.file.stitched,
                tiles.file.probability,
                tiles.file.sidebyside,
                tiles.file.segresults,
                tiles.file.error,
                tiles.file.maskraw,
                tiles.file.mask,
                tiles.file.submit,
            )

            if cfg.model.eval == 'test':
                self._validate(
                    loader=val_loader,
                    net=net,
                    force=force,
                    tiles=tiles,
                )
            elif cfg.model.eval == 'folder':
                self._validate(
                    loader=val_loader,
                    net=net,
                    criterion=criterion_val,
                    force=force,
                    tiles=tiles,
                )
            else:
                raise ValueError(f"Unknown evaluation mode: {cfg.model.eval}. ")

        return self

    def _validate(
            self,
            loader: DataLoader,
            net: torch.nn.parallel.DataParallel,
            force,
            tiles: SegTiles,
            criterion: Optional[tile2net.tiles.tileseg.loss.utils.CrossEntropyLoss2d] = None,
    ):
        """
        Run validation for one epoch
        :val_loader: data loader for validation
        """
        TILES = tiles
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

        with logging_redirect_tqdm():
            it = enumerate(tqdm(loader))
            for i, (input_images, labels, img_names, scale_float) in it:
                start = i * batch_size
                tiles = TILES.iloc[start: start + len(input_images)]
                if i % 20 == 0:
                    msg = f'Inference [Iter: {i + 1} / {len(loader)}]'
                    logger.debug(msg)

                batch = MiniBatch.from_data(
                    images=input_images,
                    net=net,
                    gt_image=labels,
                    criterion=criterion,
                    val_loss=val_loss,
                    tiles=tiles,
                    threads=threads
                )
                futures.extend(batch.submit_all())

                iou_acc += batch.iou_acc
                if (
                        cfg.options.test_mode
                        and i >= 5
                ):
                    break

        wait(futures)
        for fut in futures:
            fut.result()
        futures.clear()
