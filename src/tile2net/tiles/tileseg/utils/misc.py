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
from typing import *
from geopandas import GeoDataFrame

import tempfile
from typing import Optional

import cv2
import sys
import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils

from tabulate import tabulate
from PIL import Image

from tile2net.tiles.tileseg.config import cfg
from tile2net.namespace import Namespace
from concurrent.futures import Future, ThreadPoolExecutor

from tile2net.logger import logger

from geopandas import GeoDataFrame

if False:
    from tile2net.raster.tile import Tile
    from ...tiles import Tiles

class DumpDict(TypedDict, total=False):
    gt_images: torch.Tensor
    input_images: torch.Tensor
    predictions: np.ndarray
    err_mask: np.ndarray
    prob_mask: torch.Tensor
    img_names: List[str]
    assets: dict[str, Any]
    attn_maps: Optional[np.ndarray]




class ImageDumper:
    """
    Converts tensors to images, writes them to disk, and handles summary artifacts.
    """

    class AverageMeter:
        def __init__(self):
            self.reset()

        def reset(
                self,
        ):
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

    def __init__(
            self,
            tiles,
            val_len,
            tensorboard=True,
            write_webpage=True,
            webpage_fn='index.html',
            dump_all_images=False,  # noqa: F841  (kept for API compat)
            dump_assets=False,
            dump_for_auto_labelling=False,
            dump_for_submission=False,
            dump_num=10,
    ):
        self.tiles = tiles
        self.val_len = val_len
        self.tensorboard = tensorboard
        self.write_webpage = write_webpage
        self.webpage_fn = os.path.join(
            cfg.RESULT_DIR,
            'best_images',
            webpage_fn,
        )
        self.dump_assets = dump_assets
        self.dump_for_auto_labelling = dump_for_auto_labelling
        self.dump_for_submission = dump_for_submission
        self.viz_frequency = max(1, val_len // dump_num)

        inv_mean = [-mean / std for mean, std in zip(cfg.DATASET.MEAN, cfg.DATASET.STD)]
        inv_std = [1 / std for std in cfg.DATASET.STD]
        self.inv_normalize = standard_transforms.Normalize(
            mean=inv_mean,
            std=inv_std,
        )

        if self.dump_for_submission:
            self.save_dir = os.path.join(cfg.RESULT_DIR, 'submit')
        elif self.dump_for_auto_labelling:
            self.save_dir = os.path.join(cfg.RESULT_DIR)
        else:
            self.save_dir = os.path.join(cfg.RESULT_DIR, 'seg_results')

        self.imgs_to_tensorboard: list = []
        self.imgs_to_webpage: list = []

        self.visualize = standard_transforms.Compose(
            [
                standard_transforms.Resize(384),
                standard_transforms.CenterCrop((384, 384)),
                standard_transforms.ToTensor(),
            ],
        )
        self.args = tiles.cfg
        self.dump_percent = 100

    # ------------------------------------------------------------------ #
    #                         metric utilities                           #
    # ------------------------------------------------------------------ #
    def fast_hist(
            self,
            pred,
            gtruth,
            num_classes,
    ):
        mask = (gtruth >= 0) & (gtruth < num_classes)
        hist = np.bincount(
            num_classes * gtruth[mask].astype(int) + pred[mask],
            minlength=num_classes ** 2,
        )
        return hist.reshape(num_classes, num_classes)

    def prep_experiment(
            self,
    ):
        self.args.ngpu = torch.cuda.device_count()
        self.args.best_record = dict(mean_iu=-1, epoch=0)

    def calculate_iou(
            self,
            hist_data,
    ):
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

    def tensor_to_pil(
            self,
            img,
    ):
        inv_mean = [-m / s for m, s in zip(cfg.DATASET.MEAN, cfg.DATASET.STD)]
        inv_std = [1 / s for s in cfg.DATASET.STD]
        inv_norm = standard_transforms.Normalize(mean=inv_mean, std=inv_std)
        img = inv_norm(img).cpu()
        return standard_transforms.ToPILImage()(img).convert('RGB')

    def eval_metrics(
            self,
            iou_acc,
            net,
            optim,
            val_loss,
            epoch,
            mf_score=None,
    ):
        was_best = False

        iou_per_scale = dict()
        iou_per_scale[1.0] = iou_acc
        if self.args.distributed:
            iou_tensor = torch.cuda.FloatTensor(iou_acc)
            torch.distributed.all_reduce(iou_tensor, op=torch.distributed.ReduceOp.SUM)
            iou_per_scale[1.0] = iou_tensor.cpu().numpy()

        if self.args.global_rank != 0:
            return

        hist = iou_per_scale[self.args.default_scale]
        iu, acc, acc_cls = self.calculate_iou(hist)
        iou_per_scale = dict({self.args.default_scale: iu})

        self.print_evaluate_results(hist, iu, epoch, iou_per_scale)

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
            arch=self.args.arch,
            num_classes=cfg.DATASET_INST.num_classes,
            state_dict=net.state_dict(),
            optimizer=optim.state_dict(),
            mean_iu=mean_iu,
            command=' '.join(sys.argv[1:]),
        )
        torch.cuda.synchronize()

        if mean_iu > self.args.best_record['mean_iu']:
            was_best = True
            self.args.best_record = dict(
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
            self,
            hist,
            iu,
            epoch=0,
            iou_per_scale=None,
            eps=1e-8,
    ):
        if iou_per_scale is None:
            iou_per_scale = dict({1.0: iu})

        id2cat = cfg.DATASET_INST.trainid_to_name
        iu_FP = hist.sum(axis=1) - np.diag(hist)
        iu_FN = hist.sum(axis=0) - np.diag(hist)
        iu_TP = np.diag(hist)

        header = ['Id', 'label']
        header.extend([f'iU_{s}' for s in iou_per_scale])
        header.extend(['TP', 'FP', 'FN', 'Precision', 'Recall'])

        rows = []
        total_pix = hist.sum()

        for cid in range(len(iu)):
            rows.append(
                [
                    cid,
                    id2cat.get(cid, ''),
                    *[iou_per_scale[s][cid] * 100 for s in iou_per_scale],
                    100 * iu_TP[cid] / total_pix,
                    100 * iu_FP[cid] / total_pix,
                    100 * iu_FN[cid] / total_pix,
                    iu_TP[cid] / (iu_TP[cid] + iu_FP[cid] + eps),
                    iu_TP[cid] / (iu_TP[cid] + iu_FN[cid] + eps),
                ],
            )

        logger.debug(tabulate(rows, headers=header, floatfmt='1.2f'))

        class_names = [id2cat.get(cid, '') for cid in range(len(iu))]
        pd.DataFrame(hist, index=class_names, columns=class_names).to_csv(
            f'{cfg.RESULT_DIR}/histogram_{epoch}.csv',
        )

    def metrics_per_image(
            self,
            hist,
    ):
        FP = hist.sum(axis=1) - np.diag(hist)
        FN = hist.sum(axis=0) - np.diag(hist)
        return FP, FN

    def fmt_scale(
            self,
            prefix,
            scale,
    ):
        return f'{prefix}_{str(float(scale)).replace(".", "")}x'

    # ------------------------------------------------------------------ #
    #                        dumping utilities                            #
    # ------------------------------------------------------------------ #
    def reset(
            self,
    ):
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
            dump_dict,
            val_idx,
            testing=None,
            grid=None,
    ):
        colorize_mask_fn = cfg.DATASET_INST.colorize_mask

        for idx in range(len(dump_dict['input_images'])):
            input_image = dump_dict['input_images'][idx]
            gt_image = dump_dict['gt_images'][idx]
            prediction = dump_dict['assets']['predictions'][idx]
            img_name = dump_dict['img_names'][idx]

            er_prob, err_pil = self.save_prob_and_err_mask(
                dump_dict,
                img_name,
                idx,
                prediction,
            )

            input_image = self.inv_normalize(input_image).cpu()
            input_image = standard_transforms.ToPILImage()(input_image).convert('RGB')

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
                    input_image,
                    prediction_pil,
                    img_name,
                )

                if grid is not None:
                    idd_ = int(img_name.split('_')[-1])
                    self.save_dir = os.path.join(cfg.RESULT_DIR, 'seg_results')
                    tile = grid.tiles[grid.pose_dict[idd_]]
                    polygons = self.map_features(
                        tile,
                        np.array(prediction_pil),
                        img_array=True,
                    )
                    if polygons is not None:
                        yield polygons
            else:
                prediction_pil = colorize_mask_fn(prediction).convert('RGB')
                self.create_composite_image(
                    input_image,
                    prediction_pil,
                    img_name,
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
                dump_dict,
                img_name,
                idx,
                colorize_mask_fn,
                to_tb,
            )

        for future in self.futures:
            future.result()
        self.futures.clear()

    def create_composite_image(
            self,
            input_image,
            prediction_pil,
            img_name,
    ):
        if not self.args.dump_percent:
            return
        self.dump_percent += self.args.dump_percent
        if self.dump_percent < 100:
            return
        self.dump_percent -= 100

        os.makedirs(self.save_dir, exist_ok=True)
        composited = Image.new(
            'RGB',
            (input_image.width * 2, input_image.height),
        )
        composited.paste(input_image, (0, 0))
        composited.paste(prediction_pil, (input_image.width, 0))
        fn = os.path.join(
            self.save_dir,
            f'sidebside_{img_name}.png',
        )
        self.futures.append(
            self.threads.submit(composited.save, fn),
        )

    def get_dump_assets(
            self,
            dump_dict,
            img_name,
            idx,
            colorize_mask_fn,
            to_tensorboard,
    ):
        if not self.dump_assets:
            return
        assets = dump_dict['assets']
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
            dump_dict,
            img_name,
            idx,
            prediction,
    ):
        err_pil = None
        if 'err_mask' in dump_dict and 'prob_mask' in dump_dict['assets']:
            prob = dump_dict['assets']['prob_mask'][idx]
            err_mask = dump_dict['err_mask'][idx]  # noqa: F841
            prob_arr = (prob.cpu().numpy() * 255).astype(np.uint8)
            self.futures.append(
                self.threads.submit(
                    self.save_image,
                    prob_arr,
                    f'{img_name}_prob.png',
                ),
            )
            err_pil = Image.fromarray(prediction.astype(np.uint8)).convert('RGB')
            path = os.path.join(self.save_dir, f'{img_name}_err_mask.png')
            self.futures.append(
                self.threads.submit(err_pil.save, path),
            )
            return True, err_pil
        return False, err_pil

    @classmethod
    def map_features(
            cls,
            tile,
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
                src_img,
                class_name=name,
                class_id=cid,
                class_hole_size=hole,
                img_array=img_array,
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
            input_image,
            prediction_pil,
            img_name,
        )
