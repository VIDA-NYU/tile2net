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
import os

import numpy as np
import torch

from tile2net.logger import logger
from tile2net.tiles.tileseg.utils.misc import AverageMeter, eval_metrics
from tile2net.tiles.tileseg.utils.misc import ImageDumper
from tile2net.tiles.tileseg.utils.misc import fast_hist, fmt_scale
from tile2net.tiles.tileseg.utils.misc import metrics_per_image
from tile2net.tiles.tileseg.utils.results_page import ResultsPage

if False:
    from ...tiles import Tiles

import os
from collections import defaultdict

import numpy as np
import torch

from tile2net.logger import logger
from tile2net.tiles.tileseg.utils.misc import (
    AverageMeter,
    eval_metrics,
    ImageDumper,
    fast_hist,
    fmt_scale,
    metrics_per_image,
)
from typing import TypedDict, Any, List, Optional
import torch
import numpy as np

if False:  # type-checking
    from ...tiles import Tiles



class SegValidator:
    def __init__(self, tiles: "Tiles"):
        self.tiles = tiles
        self.cfg = tiles.cfg

    @staticmethod
    def flip_tensor(x: torch.Tensor, dim: int) -> torch.Tensor:
        dim = x.dim() + dim if dim < 0 else dim
        return x[
            tuple(
                slice(None)
                if i != dim
                else torch.arange(x.size(i) - 1, -1, -1, device=x.device).long()
                for i in range(x.dim())
            )
        ]

    def resize_tensor(self, x: torch.Tensor, target) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x,
            size=target,
            mode="bilinear",
            align_corners=self.cfg.MODEL.ALIGN_CORNERS,
        )

    def calc_err_mask(self, pred, gtruth, num_classes, classid):
        err_mask = gtruth >= 0
        err_mask &= gtruth == classid
        err_mask &= (
                ((pred == classid) & (gtruth != classid) & (gtruth != self.cfg.DATASET.IGNORE_LABEL))
                | ((pred != classid) & (gtruth == classid))
        )
        return err_mask.astype(int)

    def calc_err_mask_all(self, pred, gtruth, num_classes):
        err_mask = gtruth >= 0
        err_mask &= gtruth != self.cfg.DATASET.IGNORE_LABEL
        err_mask &= pred != gtruth
        return err_mask.astype(int)

    def eval_minibatch(
            self,
            data,
            net,
            criterion,
            val_loss,
            calc_metrics: bool,
            val_idx: int,
    ):
        args = cfg = self.cfg
        torch.cuda.empty_cache()

        scales = [args.default_scale]
        if args.multi_scale_inference:
            scales.extend(float(x) for x in args.extra_scales.split(","))
            if val_idx == 0:
                logger.debug(f"Using multi-scale inference (AVGPOOL) with scales {scales}")

        images, gt_image, img_names, scale_float = data
        batch_px = images.size(0) * images.size(2) * images.size(3)
        hw = images.size(2), images.size(3)
        flips = [1, 0] if args.do_flip else [0]

        with torch.no_grad():
            output = 0.0
            for flip in flips:
                for scale in scales:
                    inp = self.flip_tensor(images, 3) if flip else images
                    if scale != 1.0:
                        inp = self.resize_tensor(inp, [round(s * scale) for s in hw])

                    inp = {"images": inp.cuda(), "gts": gt_image.cuda()}
                    out_dict = net(inp)
                    pred = out_dict["pred"]

                    if not cfg.MODEL.MSCALE:
                        out_dict[fmt_scale("pred", scale)] = pred
                    if scale != 1.0:
                        pred = self.resize_tensor(pred, hw)

                    output += self.flip_tensor(pred, 3) if flip else pred

        output /= len(scales) * len(flips)
        gt_cuda = gt_image.cuda()
        assert output.shape[2:] == gt_cuda.shape[1:]
        assert output.shape[1] == cfg.DATASET.NUM_CLASSES

        if calc_metrics:
            val_loss.update(criterion(output, gt_cuda).item(), batch_px)

        soft = torch.nn.functional.softmax(output, dim=1).cpu()
        max_probs, predictions = soft.max(1)
        preds_np = predictions.numpy()

        assets = {
            k: (
                torch.nn.functional.softmax(v, dim=1).detach().max(1)[1].cpu().numpy()
                if "pred_" in k
                else v
            )
            for k, v in out_dict.items()
            if "pred_" in k or "attn_" in k
        }
        assets.update(
            predictions=preds_np,
            prob_mask=max_probs,
        )
        if calc_metrics:
            assets["err_mask"] = self.calc_err_mask_all(preds_np, gt_image.numpy(), cfg.DATASET.NUM_CLASSES)

        iou_acc = fast_hist(preds_np.flatten(), gt_image.numpy().flatten(), cfg.DATASET.NUM_CLASSES)
        return assets, iou_acc

    def validate_topn(
            self,
            val_loader,
            net,
            criterion,
            optim,
            epoch: int,
            dump_assets: bool = True,
            dump_all_images: bool = True,
    ):
        args = cfg = self.cfg
        assert args.bs_val == 1

        logger.debug("First pass")
        image_metrics, iou_acc = {}, 0
        dumper = ImageDumper(
            val_len=len(val_loader),
            dump_all_images=dump_all_images,
            dump_assets=dump_assets,
            dump_for_auto_labelling=args.dump_for_auto_labelling,
            dump_for_submission=args.dump_for_submission,
        )
        net.eval()
        val_loss = AverageMeter()

        for val_idx, data in enumerate(val_loader):
            assets, ia = self.eval_minibatch(
                data=data,
                net=net,
                criterion=criterion,
                val_loss=val_loss,
                calc_metrics=True,
                val_idx=val_idx,
            )

            _, _, img_names, _ = data
            fp, fn = metrics_per_image(ia)
            image_metrics[img_names[0]] = (fp, fn)
            iou_acc += ia

            if val_idx % 20 == 0:
                logger.debug(f"validating[Iter: {val_idx + 1} / {len(val_loader)}]")

            if val_idx > 5 and args.test_mode:
                break

        eval_metrics(iou_acc, args, net, optim, val_loss, epoch)

        worst_images, class_to_images = defaultdict(dict), defaultdict(dict)
        for cid in range(cfg.DATASET.NUM_CLASSES):
            tbl = {n: sum(m[i][cid] for i in (0, 1)) for n, m in image_metrics.items()}
            for n in sorted(tbl, key=tbl.get, reverse=True)[: args.dump_topn]:
                worst_images[n][cid] = tbl[n]
                class_to_images[cid][n] = tbl[n]

        logger.debug(str(worst_images))
        logger.debug("Second pass")
        attn_map = None

        for val_idx, data in enumerate(val_loader):
            in_img, gt_img, img_names, _ = data
            if not args.dump_topn_all and img_names[0] not in worst_images:
                continue

            with torch.no_grad():
                out_dict = net({"images": in_img.cuda(), "gts": gt_img})

            output = out_dict["pred"]
            prob_mask, predictions = torch.nn.functional.softmax(output, dim=1).max(1)
            assets = {
                k: torch.nn.functional.softmax(v, dim=1).detach().max(1)[1].cpu().numpy()
                for k, v in out_dict.items()
            }

            img_name = img_names[0]
            for cid, fail_px in worst_images[img_name].items():
                err_mask = self.calc_err_mask(
                    predictions.numpy(),
                    gt_img.numpy(),
                    cfg.DATASET.NUM_CLASSES,
                    cid,
                )
                cls_name = cfg.DATASET_INST.trainid_to_name[cid]
                img_cls = f"{img_name}_{cls_name}"

                dump: DumpDict = dict(
                    gt_images=gt_img,
                    input_images=in_img,
                    predictions=predictions.numpy(),
                    err_mask=err_mask,
                    prob_mask=prob_mask,
                    img_names=[img_cls],
                    assets=dict(
                        **assets,
                        predictions=predictions.numpy(),
                        prob_mask=prob_mask,
                        err_mask=err_mask,
                    ),
                    attn_maps=attn_map,
                )
                dumper.dump(dump, val_idx)

        html_fn = os.path.join(args.result_dir, "seg_results", "topn_failures.html")
        page = ResultsPage("topn failures", html_fn)
        for cid, imgs in class_to_images.items():
            cls_name = cfg.DATASET_INST.trainid_to_name[cid]
            for img_name, fail_px in sorted(imgs.items(), key=lambda x: x[1], reverse=True):
                img_cls = f"{img_name}_{cls_name}"
                page.add_table(
                    [
                        (f"{img_cls}_prediction.png", "pred"),
                        (f"{img_cls}_gt.png", "gt"),
                        (f"{img_cls}_input.png", "input"),
                        (f"{img_cls}_err_mask.png", "errors"),
                        (f"{img_cls}_prob_mask.png", "prob"),
                    ],
                    table_heading=f"{cls_name}-{fail_px}",
                )
        page.write_page()
        return val_loss.avg
