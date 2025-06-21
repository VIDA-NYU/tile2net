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


Miscellanous Functions
"""

from __future__ import annotations

from dataclasses import dataclass

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import *
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
from geopandas import GeoDataFrame
from tile2net.tiles.cfg import cfg
from tile2net.tiles.cfg.logger import logger

if False:
    from ...tiles import Tiles


def fmt_scale(
        prefix: str,
        scale: float,
) -> str:
    return f'{prefix}_{str(float(scale)).replace(".", "")}x'


def fast_hist(
        pred: np.ndarray,
        gtruth: np.ndarray,
        num_classes: int,
) -> np.ndarray:
    mask = (gtruth >= 0) & (gtruth < num_classes)
    hist = np.bincount(
        num_classes * gtruth[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2,
    )
    return hist.reshape(num_classes, num_classes)


def calculate_iou(
        hist_data: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    if (
            not isinstance(hist_data, np.ndarray)
            or len(hist_data.shape) != 2
            or hist_data.shape[0] != hist_data.shape[1]
    ):
        raise ValueError('hist_data must be a square 2D numpy array')

    acc = np.diag(hist_data).sum() / hist_data.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist_data) / hist_data.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        divisor = (
                hist_data.sum(axis=1)
                + hist_data.sum(axis=0)
                - np.diag(hist_data)
        )
        iu = np.diag(hist_data) / divisor
        iu = iu[~np.isnan(iu)]

    return iu, acc, acc_cls


def prep_experiment(
) -> None:
    cfg.ngpu = torch.cuda.device_count()
    cfg.best_record = dict(mean_iu=-1, epoch=0)


def tensor_to_pil(
        img: torch.Tensor,
) -> Image.Image:
    inv_mean = [-m / s for m, s in zip(cfg.dataset.mean, cfg.dataset.std)]
    inv_std = [1 / s for s in cfg.dataset.std]
    inv_norm = standard_transforms.Normalize(mean=inv_mean, std=inv_std)
    img = inv_norm(img).cpu()
    return standard_transforms.ToPILImage()(img).convert('RGB')


def eval_metrics(
        iou_acc: np.ndarray,
        net: torch.nn.Module,
        optim: torch.optim.Optimizer,
        val_loss: AverageMeter,
        epoch: int,
        mf_score: Optional[float] = None,
) -> None:
    was_best = False

    iou_per_scale = dict()
    iou_per_scale[1.0] = iou_acc
    if cfg.distributed:
        iou_tensor = torch.cuda.FloatTensor(iou_acc)
        torch.distributed.all_reduce(iou_tensor, op=torch.distributed.ReduceOp.SUM)
        iou_per_scale[1.0] = iou_tensor.cpu().numpy()

    if cfg.global_rank != 0:
        return

    hist = iou_per_scale[cfg.default_scale]
    iu, acc, acc_cls = calculate_iou(hist)
    iou_per_scale = dict({cfg.default_scale: iu})

    print_evaluate_results(hist, iu, epoch, iou_per_scale)

    freq = hist.sum(axis=1) / hist.sum()
    mean_iu = np.nanmean(iu)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    _ = dict(
        loss=val_loss.avg,
        mean_iu=mean_iu,
        acc_cls=acc_cls,
        acc=acc,
    )  # metrics for potential logging

    save_dict = dict(
        epoch=epoch,
        arch=cfg.arch,
        num_classes=cfg.dataset_inst.num_classes,
        state_dict=net.state_dict(),
        optimizer=optim.state_dict(),
        mean_iu=mean_iu,
        command=' '.join(sys.argv[1:]),
    )
    torch.cuda.synchronize()

    if mean_iu > cfg.best_record['mean_iu']:
        was_best = True
        cfg.best_record = dict(
            val_loss=val_loss.avg,
            mask_f1_score=mf_score.avg if mf_score else None,
            acc=acc,
            acc_cls=acc_cls,
            fwavacc=fwavacc,
            mean_iu=mean_iu,
            epoch=epoch,
        )

    logger.debug('-' * 107)
    return was_best


def print_evaluate_results(
        hist: np.ndarray,
        iu: np.ndarray,
        epoch: int = 0,
        iou_per_scale: Optional[dict[float, np.ndarray]] = None,
        eps: float = 1e-8,
) -> None:
    if iou_per_scale is None:
        iou_per_scale = dict({1.0: iu})

    id2cat = cfg.dataset_inst.trainid_to_name
    iu_FP = hist.sum(axis=1) - np.diag(hist)
    iu_FN = hist.sum(axis=0) - np.diag(hist)
    iu_TP = np.diag(hist)

    header = ['Id', 'label']
    header.extend([f'iU_{s}' for s in iou_per_scale])
    header.extend(['TP', 'FP', 'FN', 'Precision', 'Recall'])

    rows = []
    total_pix = hist.sum()

    for cid in range(len(iu)):
        item = [
            cid,
            id2cat.get(cid, ''),
            *[iou_per_scale[s][cid] * 100 for s in iou_per_scale],
            100 * iu_TP[cid] / total_pix,
            100 * iu_FP[cid] / total_pix,
            100 * iu_FN[cid] / total_pix,
            iu_TP[cid] / (iu_TP[cid] + iu_FP[cid] + eps),
            iu_TP[cid] / (iu_TP[cid] + iu_FN[cid] + eps),
        ]
        rows.append(item)

    logger.debug(tabulate(rows, headers=header, floatfmt='1.2f'))

    class_names = [id2cat.get(cid, '') for cid in range(len(iu))]
    pd.DataFrame(hist, index=class_names, columns=class_names).to_csv(
        f'{cfg.result_dir}/histogram_{epoch}.csv',
    )


def metrics_per_image(
        hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    FP = hist.sum(axis=1) - np.diag(hist)
    FN = hist.sum(axis=0) - np.diag(hist)
    return FP, FN


@dataclass
class DumpData:
    gt_images: torch.Tensor
    input_images: torch.Tensor
    # img_names: List[str]
    assets: dict[str, Any]
    predictions: np.ndarray = None
    err_mask: np.ndarray = None
    prob_mask: torch.Tensor = None
    attn_maps: Optional[np.ndarray] = None

    error_files: np.ndarray = None
    prob_files: np.ndarray = None
    sidebyside_files: np.ndarray = None


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(
            self,
    ) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
            self,
            val,
            n=1,
    ):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ImageDumper:
    """
    Converts tensors to images, writes them to disk, and handles summary artifacts.
    """

    def __init__(
            self,
            tiles: Tiles,
            val_len,
            tensorboard=True,
            write_webpage=True,
            webpage_fn='index.html',
            dump_all_images=False,  # noqa: F841  (kept for API compat)
            dump_assets=False,
            dump_for_auto_labelling=False,
            dump_for_submission=False,
            dump_num=10,
            *args,
            **kwargs
    ):
        self.tiles = tiles
        self.val_len = val_len
        self.tensorboard = tensorboard
        self.write_webpage = write_webpage
        self.webpage_fn = tiles.outdir.best_images.webpage
        self.dump_assets = dump_assets
        self.dump_for_auto_labelling = dump_for_auto_labelling
        self.dump_for_submission = dump_for_submission
        self.viz_frequency = max(1, val_len // dump_num)

        inv_mean = [-mean / std for mean, std in zip(cfg.dataset.mean, cfg.dataset.std)]
        inv_std = [1 / std for std in cfg.dataset.std]
        self.inv_normalize = standard_transforms.Normalize(
            mean=inv_mean,
            std=inv_std,
        )

        if self.dump_for_submission:
            self.save_dir = tiles.outdir.submit.dir
        elif self.dump_for_auto_labelling:
            self.save_dir = tiles.outdir.dir
        else:
            self.save_dir = tiles.outdir.seg_results.dir

        self.imgs_to_tensorboard: list = []
        self.imgs_to_webpage: list = []

        self.visualize = standard_transforms.Compose(
            [
                standard_transforms.Resize(384),
                standard_transforms.CenterCrop((384, 384)),
                standard_transforms.ToTensor(),
            ],
        )
        self.dump_percent = 100

    def reset(self):
        self.imgs_to_tensorboard.clear()
        self.imgs_to_webpage.clear()

    def save_image(
            self,
            image,
            filename,
    ):
        cv2.imwrite(os.path.join(self.save_dir, filename), image)

    # (existing dump / create_composite_image / asset helpers follow with the
    #  same API, making sure every parameter occupies its own line and any
    #  literal dicts are rewritten with dict(key=value, …) style.)


class ThreadedDumper(
    ImageDumper,
):
    def __init__(
            self,
            *args,
            max_workers: int | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.threads: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=max_workers,
        )
        self.futures: list[Future] = []

    def dump(
            self,
            dump_data: DumpData,
            val_idx,
            testing=None,
            tiles: Tiles = None,
    ):
        colorize_mask_fn = cfg.dataset_inst.colorize_mask

        for idx in range(len(dump_data.input_images)):
            input_image = dump_data.input_images[idx]
            gt_image = dump_data.gt_images[idx]
            prediction = dump_data.assets['predictions'][idx]
            sidebyside = dump_data.sidebyside_files[idx]
            prob = dump_data.prob_files[idx]
            error = dump_data.error_files[idx]

            er_prob, err_pil = self.save_prob_and_err_mask(
                dump_data=dump_data,
                idx=idx,
                prediction=prediction,
                prob_file=prob,
                err_file=error,
            )

            # input_image = standard_transforms.ToPILImage()(input_image).convert('RGB')
            input_image = (
                standard_transforms
                .ToPILImage()(input_image)
                .convert('RGB')
            )

            if testing:
                alpha = False
                all_pix = np.prod(np.array(input_image).shape[:2])
                if np.array(input_image).shape[-1] == 3:
                    black = np.count_nonzero(
                        np.all(np.array(input_image) == [0, 0, 0], axis=2),
                    )
                else:
                    black = np.count_nonzero(
                        np.all(np.array(input_image) == [0, 0, 0, 0], axis=-1),
                    )
                    alpha = True
                if black / all_pix > 0.25 or alpha:
                    continue

                prediction_pil = colorize_mask_fn(prediction).convert('RGB')
                self.create_composite_image(
                    input_image=input_image,
                    prediction_pil=prediction_pil,
                    sidebyside=sidebyside
                )

                if tiles is not None:
                    # idd_ = int(img_name.split('_')[-1])
                    # self.save_dir = tiles.outdir.seg_results.dir
                    polygons = self.map_features(
                        src_img=np.array(prediction_pil),
                        img_array=True
                    )
                    if polygons is not None:
                        yield polygons
            else:
                prediction_pil = colorize_mask_fn(prediction).convert('RGB')
                self.create_composite_image(
                    input_image=input_image,
                    prediction_pil=prediction_pil,
                )

            gt_pil = colorize_mask_fn(gt_image.cpu().numpy())
            to_tb = [
                self.visualize(input_image),
                self.visualize(gt_pil.convert('RGB')),
                self.visualize(prediction_pil),
            ]
            if er_prob and err_pil is not None:
                to_tb.append(self.visualize(err_pil.convert('RGB')))

            self.get_dump_assets(
                dump_dict=dump_dict,
                img_name=img_name,
                idx=idx,
                colorize_mask_fn=colorize_mask_fn,
                to_tensorboard=to_tb
            )

        for future in self.futures:
            future.result()
        self.futures.clear()

    def create_composite_image(
            self,
            input_image,
            prediction_pil,
            sidebyside: str,
    ):
        if not cfg.dump_percent:
            return
        self.dump_percent += cfg.dump_percent
        if self.dump_percent < 100:
            return
        self.dump_percent -= 100

        os.makedirs(self.save_dir, exist_ok=True)

        size = (input_image.width * 2, input_image.height)
        composited = Image.new('RGB', size)
        composited.paste(input_image, (0, 0))
        composited.paste(prediction_pil, (input_image.width, 0))
        # fn = os.path.join(
        #     self.save_dir,
        #     f'sidebside_{img_name}.png',
        # )
        self.futures.append(
            self.threads.submit(composited.save, sidebyside),
        )

    def get_dump_assets(
            self,
            dump_data: DumpData,
            img_name,
            idx,
            colorize_mask_fn,
            to_tensorboard,
    ):
        if not self.dump_assets:
            return
        assets = dump_data.assets
        for asset, mask in assets.items():
            mask = mask[idx]
            fn = os.path.join(
                self.save_dir,
                f'{img_name}_{asset}.png',
            )
            if asset.startswith('pred_'):
                pred_pil = colorize_mask_fn(mask)
                self.futures.append(
                    self.threads.submit(pred_pil.save, fn),
                )
                continue
            mask_arr = (
                mask.squeeze().cpu().numpy()
                if isinstance(mask, torch.Tensor) else mask.squeeze()
            )
            mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8)).convert('RGB')
            self.futures.append(
                self.threads.submit(mask_pil.save, fn),
            )
            to_tensorboard.append(self.visualize(mask_pil))

    def save_prob_and_err_mask(
            self,
            dump_data: DumpData,
            # img_name,
            idx,
            prediction,
            prob_file,
            err_file,
    ):
        err_pil = None
        # Check if err_mask and prob_mask are available
        if (
                dump_data.err_mask is not None
                and 'prob_mask' in dump_data.assets
        ):
            prob = dump_data.assets['prob_mask'][idx]
            err_mask = dump_data.err_mask[idx]  # noqa: F841
            prob_arr = (prob.cpu().numpy() * 255).astype(np.uint8)
            future = self.threads.submit(self.save_image, prob_arr, prob_file)
            self.futures.append(future)
            err_pil = Image.fromarray(prediction.astype(np.uint8)).convert('RGB')
            future = self.threads.submit(err_pil.save, err_file)
            self.futures.append(future)
            return True, err_pil
        return False, err_pil

    @classmethod
    def map_features(
            cls,
            src_img: np.ndarray,
            img_array=True,
    ) -> GeoDataFrame | None:
        layers: list[GeoDataFrame] = []
        for name, cid, hole in [
            ('sidewalk', 2, None),
            ('crosswalk', 0, 15),
            ('road', 1, 30),
        ]:
            gdf = tile.mask2poly(
                src_img=src_img,
                class_name=name,
                class_id=cid,
                class_hole_size=hole,
                img_array=img_array
            )
            if gdf is not False:
                layers.append(gdf)
        if layers:
            geoms = pd.concat(layers).reset_index(drop=True)
            return geoms


class PushDumper(
    ThreadedDumper,
):
    def create_composite_image(
            self,
            input_image,
            prediction_pil,
            img_name: str,
    ):
        return super().create_composite_image(
            input_image=input_image,
            prediction_pil=prediction_pil,
            img_name=img_name
        )
