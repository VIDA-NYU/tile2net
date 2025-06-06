import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import *
import torch
import numpy as np
import pandas as pd

from tile2net.tiles.cfg import cfg
from tile2net.tileseg.utils.misc import fast_hist, fmt_scale
from tile2net.tileseg.utils.misc import AverageMeter, eval_metrics
from tile2net.tileseg.utils.misc import metrics_per_image
from tile2net.tileseg.utils.misc import ImageDumper
from tile2net.logger import logger
from dataclasses import dataclass
from typing import Self, Optional


@dataclass
class MiniBatch(

):
    predictions: np.ndarray
    prob_mask: torch.Tensor
    iou_acc: np.ndarray
    pred_05x: Optional[np.ndarray] = None
    pred_10x: Optional[np.ndarray] = None
    attn_05x: Optional[np.ndarray] = None
    attn_10x: Optional[np.ndarray] = None

    @classmethod
    def from_data(
            cls,
            data: tuple,
            net: torch.nn.Module,
            criterion: torch.nn.Module,
            val_loss: AverageMeter,
            calc_metrics: bool,
            val_idx: int
    ):
        """
        Evaluate a single minibatch of images.
         * calculate metrics
         * dump images
        There are two primary multi-scale inference types:
          1. 'MSCALE', or in-model multi-scale: where the multi-scale iteration loop is
             handled within the model itself (see networks/mscale.py -> nscale_forward())
          2. 'multi_scale_inference', where we use Averaging to combine scales
        """
        torch.cuda.empty_cache()

        scales = [cfg.default_scale]
        if cfg.multi_scale_inference:
            it = (
                float(x)
                for x in cfg.extra_scales.split(',')
            )
            scales.extend(it)
            if val_idx == 0:
                logger.debug(f'Using multi-scale inference (AVGPOOL) with scales {scales}')

        # input    = torch.Size([1, 3, h, w])
        # gt_image = torch.Size([1, h, w])
        images, gt_image, img_names, scale_float = data
        assert len(images.size()) == 4 and len(gt_image.size()) == 3
        assert images.size()[2:] == gt_image.size()[1:]
        batch_pixel_size = images.size(0) * images.size(2) * images.size(3)
        input_size = images.size(2), images.size(3)

        if cfg.do_flip:
            # By ending with flip=0, we insure that the images that are dumped
            # out correspond to the unflipped versions. A bit hacky.
            flips = [1, 0]
        else:
            flips = [0]

        with torch.no_grad():
            output = 0.0

            for flip in flips:
                for scale in scales:
                    if flip == 1:
                        inputs = cls.flip_tensor(images, 3)
                    else:
                        inputs = images

                    infer_size = [round(sz * scale) for sz in input_size]

                    if scale != 1.0:
                        inputs = cls.resize_tensor(inputs, infer_size)

                    inputs = {'images': inputs, 'gts': gt_image}
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                    # Expected Model outputs:
                    #   required:
                    #     'pred'  the network prediction, shape (1, 19, h, w)
                    #
                    #   optional:
                    #     'pred_*' - multi-scale predictions from mscale model
                    #     'attn_*' - multi-scale attentions from mscale model
                    output_dict = net(inputs)

                    _pred = output_dict['pred']

                    # save AVGPOOL style multi-scale output for visualizing
                    if not cfg.MODEL.MSCALE:
                        scale_name = fmt_scale('pred', scale)
                        output_dict[scale_name] = _pred

                    # resize tensor down to 1.0x scale in order to combine
                    # with other scales of prediction
                    if scale != 1.0:
                        _pred = cls.resize_tensor(_pred, input_size)

                    if flip == 1:
                        output = output + cls.flip_tensor(_pred, 3)
                    else:
                        output = output + _pred

        output = output / len(scales) / len(flips)
        assert_msg = 'output_size {} gt_cuda size {}'
        gt_cuda = gt_image.cuda()
        assert_msg = assert_msg.format(
            output.size()[2:],
            gt_cuda.size()[1:]
        )
        assert output.size()[2:] == gt_cuda.size()[1:], assert_msg
        assert output.size()[1] == cfg.DATASET.NUM_CLASSES, assert_msg

        # Update loss and scoring datastructure
        if calc_metrics:
            val_loss.update(
                criterion(output, gt_image.cuda()).item(),
                batch_pixel_size
            )

        output_data = torch.nn.functional.softmax(output, dim=1).cpu().data
        max_probs, predictions = output_data.max(1)

        # Assemble assets to visualize
        assets = {}
        for item in output_dict:
            if 'attn_' in item:
                assets[item] = output_dict[item]
            if 'pred_' in item:
                # computesoftmax for each class
                smax = torch.nn.functional.softmax(output_dict[item], dim=1)
                _, pred = smax.detach().max(1)
                assets[item] = pred.cpu().numpy()

        predictions = predictions.numpy()
        assets['predictions'] = predictions
        assets['prob_mask'] = max_probs
        if calc_metrics:
            assets['err_mask'] = cls.calc_err_mask_all(
                predictions,
                gt_image.numpy(),
                cfg.DATASET.NUM_CLASSES
            )

        _iou_acc = fast_hist(
            predictions.flatten(),
            gt_image.numpy().flatten(),
            cfg.DATASET.NUM_CLASSES
        )
        result = cls(
            predictions=predictions,
            prob_mask=max_probs,
            iou_acc=_iou_acc,
            pred_05x=assets.get('pred_05x', None),
            pred_10x=assets.get('pred_10x', None),
            attn_05x=assets.get('attn_05x', None),
            attn_10x=assets.get('attn_10x', None),
        )
        return result

        # return assets, _iou_acc

    @classmethod
    def flip_tensor(
            cls,
            x: torch.Tensor,
            dim: int
    ) -> torch.Tensor:
        """
        Flip Tensor along a dimension
        """
        dim = x.dim() + dim if dim < 0 else dim
        # return x[tuple(slice(None, None) if i != dim
        #                else torch.arange(x.size(i) - 1, -1, -1).long()
        #                for i in range(x.dim()))]
        item = tuple(
            slice(None, None) if i != dim
            else torch.arange(x.size(i) - 1, -1, -1).long()
            for i in range(x.dim())
        )
        return x[item]

    @classmethod
    def resize_tensor(
            cls,
            inputs: torch.Tensor,
            target_size: tuple[int, int]
    ) -> torch.Tensor:
        inputs = torch.nn.functional.interpolate(
            inputs,
            size=target_size,
            mode='bilinear',
            align_corners=cfg.MODEL.ALIGN_CORNERS
        )
        return inputs

    @classmethod
    def calc_err_mask(
            cls,
            pred: np.ndarray,
            gtruth: np.ndarray,
            num_classes: int,
            classid: int
    ) -> np.ndarray:
        """
        calculate class-specific error masks
        """
        # Class-specific error mask
        class_mask = (gtruth >= 0) & (gtruth == classid)
        fp = (pred == classid) & ~class_mask & (gtruth != cfg.DATASET.IGNORE_LABEL)
        fn = (pred != classid) & class_mask
        err_mask = fp | fn

        return err_mask.astype(int)

    @classmethod
    def calc_err_mask_all(
            cls,
            pred: np.ndarray,
            gtruth: np.ndarray,
            num_classes: int
    ) -> np.ndarray:
        """
        calculate class-agnostic error masks
        """
        # Class-specific error mask
        mask = (gtruth >= 0) & (gtruth != cfg.DATASET.IGNORE_LABEL)
        err_mask = mask & (pred != gtruth)

        return err_mask.astype(int)

    def to_error(
            self,
            files: pd.Series
    ) -> Iterator[Future]:
        ...

    def to_sidebyside(
            self,
            files: pd.Series
    ) -> Iterator[Future]:
        ...

    def to_probability(
            self,
            files: pd.Series
    ) -> Iterator[Future]:
        ...

    def to_mask(
            self,
            files: pd.Series
    ) -> Iterator[Future]:
        ...

    def to_mask_raw(
            self,
            files: pd.Series
    ) -> Iterator[Future]:
        ...

    def to_polygons(
            self,
            files: pd.Series,
            affile: pd.Series,
    ) -> Iterator[Future]:
        ...
