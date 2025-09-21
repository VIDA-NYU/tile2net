from __future__ import annotations
from contextlib import contextmanager
from functools import singledispatchmethod, singledispatch

import ctypes
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import *
from typing import Optional

import cv2
import numpy as np
import torch

from tile2net.grid.cfg import cfg
from tile2net.grid.cfg.logger import logger
from tile2net.tileseg.utils.misc import fast_hist, fmt_scale
from .submit import Submit
from ..grid.grid import Grid

cv2.setNumThreads(1)

if False:
    from .seggrid import SegGrid
    from tile2net.grid import Grid
@contextmanager
def maybe_autocast(
        enabled: bool,
        dtype: torch.dtype = torch.float16,
):
    # prefer new API if present; otherwise fall back
    if not enabled:
        yield
        return
    if (
            hasattr(torch, "amp")
            and hasattr(torch.amp, "autocast")
    ):
        with torch.amp.autocast("cuda", dtype=dtype):
            yield
    else:
        # older PyTorch
        with torch.cuda.amp.autocast(dtype=dtype):
            yield


def to_numpy(obj: Any):
    """converts tensors to ndarrays; preserves lists, dicts, etc."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(to_numpy(v) for v in obj)
    return obj

@singledispatch
def clip_image(
        array,
        clip: int,
):
    raise TypeError(f"Unsupported type for clip_image: {type(array)!r}")


@clip_image.register
def _clip_image_np(
        array: np.ndarray,
        clip: int,
) -> np.ndarray:
    if clip <= 0:
        return array

    nd = array.ndim

    if nd == 2:
        h, w = array.shape
        assert h > 2 * clip and w > 2 * clip, (
            f"Invalid clip={clip} for shape {array.shape}"
        )
        view = array[clip:-clip, clip:-clip]

    elif nd == 3:
        # assume channel-last (H, W, C)
        h, w = array.shape[0], array.shape[1]
        assert h > 2 * clip and w > 2 * clip, (
            f"Invalid clip={clip} for shape {array.shape}"
        )
        view = array[clip:-clip, clip:-clip, :]

    elif nd == 4:
        # support both (N, C, H, W) and (N, H, W, C)
        # decide by which axes look like spatial
        n0, n1, n2, n3 = array.shape
        # if channel-first -> (N,C,H,W): spatial = (-2,-1)
        if n2 >= 2 * clip + 1 and n3 >= 2 * clip + 1:
            view = array[:, :, clip:-clip, clip:-clip]
        # if channel-last -> (N,H,W,C): spatial = (1,2)
        elif n1 >= 2 * clip + 1 and n2 >= 2 * clip + 1:
            view = array[:, clip:-clip, clip:-clip, :]
        else:
            raise AssertionError(f"Invalid clip={clip} for shape {array.shape}")

    else:
        # leave uncommon ranks unchanged
        return array

    return np.ascontiguousarray(view)


@clip_image.register
def _clip_image_torch(
        array: torch.Tensor,
        clip: int,
) -> torch.Tensor:
    if clip <= 0:
        return array

    nd = array.ndim

    if nd == 2:
        h, w = array.shape
        assert h > 2 * clip and w > 2 * clip, (
            f"Invalid clip={clip} for shape {tuple(array.shape)}"
        )
        view = array[clip:-clip, clip:-clip]

    elif nd == 3:
        # treat as (..., H, W) if it’s actually CHW or NHW?
        # be explicit: for 3D tensors we assume channel-first (C,H,W)
        h, w = array.shape[-2], array.shape[-1]
        assert h > 2 * clip and w > 2 * clip, (
            f"Invalid clip={clip} for shape {tuple(array.shape)}"
        )
        view = array[..., clip:-clip, clip:-clip]

    elif nd == 4:
        # common case: (N, C, H, W) -> clip last two dims
        h, w = array.shape[-2], array.shape[-1]
        assert h > 2 * clip and w > 2 * clip, (
            f"Invalid clip={clip} for shape {tuple(array.shape)}"
        )
        view = array[..., clip:-clip, clip:-clip]

    else:
        return array

    return view.contiguous()


@dataclass
class MiniBatch(

):
    input_images: torch.Tensor
    predictions: np.ndarray
    iou_acc: np.ndarray
    gt_images: torch.Tensor
    grid: SegGrid
    submit: Submit

    output: dict[str, torch.Tensor] = field(default_factory=dict)
    prob_mask: Optional[torch.Tensor] = None
    error_mask: Optional[np.ndarray] = None
    clip: int = 0

    @classmethod
    def from_data(
            cls,
            images,
            gt_image,
            net: torch.nn.Module,
            grid: Grid,
            submit: Submit,
            clip: int = 0,
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
        calc_metrics = cfg.calc_metrics

        scales = [cfg.default_scale]
        if cfg.multi_scale_inference:
            it = (
                float(x)
                for x in cfg.model.extra_scales.split(',')
            )
            scales.extend(it)

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

                    infer_size = [
                        round(sz * scale)
                        for sz in input_size
                    ]

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
        gt_cuda = gt_image.cuda()
        assert_msg = (
            f'output_size {output.size()[2:]} '
            f'gt_cuda size {gt_cuda.size()[1:]}'
        )
        assert output.size()[2:] == gt_cuda.size()[1:], assert_msg
        assert output.size()[1] == cfg.DATASET.NUM_CLASSES, assert_msg

        output_data = (
            torch.nn.functional
            .softmax(output, dim=1)
            .cpu()
            .data
        )
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
        err_mask = None

        _iou_acc = fast_hist(
            predictions.flatten(),
            gt_image.numpy().flatten(),
            cfg.DATASET.NUM_CLASSES
        )
        result = cls(
            predictions=predictions,
            prob_mask=max_probs,
            iou_acc=_iou_acc,
            grid=grid,
            error_mask=err_mask,
            input_images=images,
            gt_images=gt_image,
            # threads=threads,
            submit=submit,
            clip=clip,
        )
        return result

    @classmethod
    def from_data(
            cls,
            images: torch.Tensor,
            gt_image: torch.Tensor,
            net: torch.nn.Module,
            grid: Grid,
            submit: Submit,
            clip: int = 0,
    ):

        # scales & flips
        scales = [cfg.default_scale]
        if cfg.multi_scale_inference:
            scales.extend(
                float(x)
                for x in cfg.model.extra_scales.split(',')
            )

        assert len(images.size()) == 4 and len(gt_image.size()) == 3
        assert images.size()[2:] == gt_image.size()[1:]
        input_size = images.size(2), images.size(3)

        if cfg.do_flip:
            flips = 1, 0
        else:
            flips = 0,

        with torch.no_grad():
            output = 0.0

            for flip in flips:
                for scale in scales:
                    if flip == 1:
                        inputs = cls.flip_tensor(images, 3)
                    else:
                        inputs = images
                    infer_size = [
                        round(sz * scale)
                        for sz in input_size
                    ]
                    if scale != 1.0:
                        inputs = cls.resize_tensor(inputs, infer_size)

                    inputs = dict(
                        images=inputs.cuda(non_blocking=True),
                        gts=gt_image.cuda(non_blocking=True),
                    )

                    output_dict = net(inputs)
                    _pred = output_dict["pred"]

                    if not cfg.MODEL.MSCALE:
                        scale_name = fmt_scale("pred", scale)
                        output_dict[scale_name] = _pred

                    if scale != 1.0:
                        _pred = cls.resize_tensor(_pred, input_size)

                    if flip == 1:
                        output += cls.flip_tensor(_pred, 3)
                    else:
                        output += _pred

            # average over flips/scales
            output = output / (len(scales) * len(flips))

            # GPU-side postproc: softmax, argmax, quantize
            probs = torch.softmax(output, dim=1)
            max_probs, preds = probs.max(1)

            preds_u8 = preds.to(torch.uint8)  # (N,H,W)
            prob_u8 = (
                max_probs.mul_(255.0)
                .clamp_(0, 255)
                .to(torch.uint8)
            )

        # move only minimal artifacts to CPU for metrics / saving
        predictions_np: np.ndarray = preds_u8.cpu().numpy()

        # IoU on CPU against CPU ground-truth
        _iou_acc = fast_hist(
            predictions_np.flatten(),
            gt_image.numpy().flatten(),
            cfg.DATASET.NUM_CLASSES,
        )

        # assemble optional assets (avoid recursive moves)
        assets = {}
        if cfg.get("dump_assets", False):
            for key, val in output_dict.items():
                if key.startswith("attn_"):
                    assets[key] = val  # keep on device; downstream decides when to move
                if key.startswith("pred_"):
                    smax = torch.softmax(val, dim=1)
                    _, pred_k = smax.detach().max(1)
                    assets[key] = pred_k.cpu().numpy()

        result = cls(
            predictions=predictions_np,
            prob_mask=prob_u8 if cfg.segmentation.probability else None,  # stays GPU until submit
            iou_acc=_iou_acc,
            grid=grid,
            error_mask=None,
            input_images=images,
            gt_images=gt_image,
            submit=submit,
            clip=clip,
        )
        # optional: attach assets dict if your class expects/uses it
        result.output.update(assets)
        return result

    @classmethod
    def from_data(
            cls,
            images: torch.Tensor,
            gt_image: torch.Tensor,
            net: torch.nn.Module,
            grid: Grid,
            submit: Submit,
            clip: int = 0,
    ):

        # scales & flips
        scales = [cfg.default_scale]
        if cfg.multi_scale_inference:
            scales.extend(
                float(x)
                for x in cfg.model.extra_scales.split(',')
            )

        assert len(images.size()) == 4 and len(gt_image.size()) == 3
        assert images.size()[2:] == gt_image.size()[1:]
        input_size = images.size(2), images.size(3)

        if cfg.do_flip:
            flips = 1, 0
        else:
            flips = 0,

        with torch.no_grad():
            output = 0.0

            for flip in flips:
                for scale in scales:
                    if flip == 1:
                        inputs = cls.flip_tensor(images, 3)
                    else:
                        inputs = images
                    infer_size = [
                        round(sz * scale)
                        for sz in input_size
                    ]
                    if scale != 1.0:
                        inputs = cls.resize_tensor(inputs, infer_size)

                    inputs = dict(
                        images=inputs.cuda(non_blocking=True),
                        gts=gt_image.cuda(non_blocking=True),
                    )

                    output_dict = net(inputs)
                    _pred = output_dict["pred"]

                    if not cfg.MODEL.MSCALE:
                        scale_name = fmt_scale("pred", scale)
                        output_dict[scale_name] = _pred

                    if scale != 1.0:
                        _pred = cls.resize_tensor(_pred, input_size)

                    if flip == 1:
                        output += cls.flip_tensor(_pred, 3)
                    else:
                        output += _pred

            # average over flips/scales
            output = output / (len(scales) * len(flips))

            # GPU-side postproc: softmax, argmax, quantize
            probs = torch.softmax(output, dim=1)
            max_probs, preds = probs.max(1)

            # preds_u8 = preds.to(torch.uint8)  # (N,H,W)
            predictions = (
                preds
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            if cfg.segmentation.probability:
                prob_mask = (
                    max_probs.mul_(255.0)
                    .clamp_(0, 255)
                    .to(torch.uint8)
                )
            else:
                prob_mask = None

        # move only minimal artifacts to CPU for metrics / saving

        # IoU on CPU against CPU ground-truth
        _iou_acc = fast_hist(
            predictions.flatten(),
            gt_image.numpy().flatten(),
            cfg.DATASET.NUM_CLASSES,
        )

        # assemble optional assets (avoid recursive moves)
        assets = {}
        if cfg.get("dump_assets", False):
            for key, val in output_dict.items():
                if key.startswith("attn_"):
                    assets[key] = val  # keep on device; downstream decides when to move
                if key.startswith("pred_"):
                    assets[key] = (
                        torch.softmax(val, dim=1)
                        .detach()
                        .max(1)
                        [1]
                        .cpu()
                        .numpy()
                    )

        result = cls(
            predictions=predictions,
            prob_mask=prob_mask,
            iou_acc=_iou_acc,
            grid=grid,
            error_mask=None,
            input_images=images,
            gt_images=gt_image,
            submit=submit,
            clip=clip,
        )
        # optional: attach assets dict if your class expects/uses it
        result.output.update(assets)
        return result


    @classmethod
    def from_data(
            cls,
            images: torch.Tensor,
            gt_image: torch.Tensor,
            net: torch.nn.Module,
            grid: Grid,
            submit: Submit,
            clip: int = 0,
    ):

        # scales & flips
        scales = [cfg.default_scale]
        if cfg.multi_scale_inference:
            scales.extend(
                float(x)
                for x in cfg.model.extra_scales.split(',')
            )

        assert len(images.size()) == 4 and len(gt_image.size()) == 3
        assert images.size()[2:] == gt_image.size()[1:]
        input_size = images.size(2), images.size(3)

        if cfg.do_flip:
            flips = 1, 0
        else:
            flips = 0,

        with torch.no_grad():
            output = 0.0

            for flip in flips:
                for scale in scales:
                    if flip == 1:
                        inputs = cls.flip_tensor(images, 3)
                    else:
                        inputs = images
                    infer_size = [
                        round(sz * scale)
                        for sz in input_size
                    ]
                    if scale != 1.0:
                        inputs = cls.resize_tensor(inputs, infer_size)

                    inputs = dict(
                        images=inputs.cuda(non_blocking=True),
                        gts=gt_image.cuda(non_blocking=True),
                    )

                    output_dict = net(inputs)
                    _pred = output_dict["pred"]

                    if not cfg.MODEL.MSCALE:
                        scale_name = fmt_scale("pred", scale)
                        output_dict[scale_name] = _pred

                    if scale != 1.0:
                        _pred = cls.resize_tensor(_pred, input_size)

                    if flip == 1:
                        output += cls.flip_tensor(_pred, 3)
                    else:
                        output += _pred

            # average over flips/scales
            output = output / (len(scales) * len(flips))

            # GPU-side postproc: softmax, argmax, quantize
            probs = torch.softmax(output, dim=1)
            max_probs, preds = probs.max(1)

            # preds_u8 = preds.to(torch.uint8)  # (N,H,W)
            predictions = (
                preds
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            if cfg.segmentation.probability:
                prob_mask = (
                    max_probs.mul_(255.0)
                    .clamp_(0, 255)
                    .to(torch.uint8)
                )
            else:
                prob_mask = None

        # move only minimal artifacts to CPU for metrics / saving

        # IoU on CPU against CPU ground-truth
        _iou_acc = fast_hist(
            predictions.flatten(),
            gt_image.numpy().flatten(),
            cfg.DATASET.NUM_CLASSES,
        )

        # assemble optional assets (avoid recursive moves)
        assets = {}
        if cfg.get("dump_assets", False):
            for key, val in output_dict.items():
                if key.startswith("attn_"):
                    assets[key] = val  # keep on device; downstream decides when to move
                if key.startswith("pred_"):
                    assets[key] = (
                        torch.softmax(val, dim=1)
                        .detach()
                        .max(1)
                        [1]
                        .cpu()
                        .numpy()
                    )

        result = cls(
            predictions=predictions,
            prob_mask=prob_mask,
            iou_acc=_iou_acc,
            grid=grid,
            error_mask=None,
            input_images=images,
            gt_images=gt_image,
            submit=submit,
            clip=clip,
        )
        # optional: attach assets dict if your class expects/uses it
        result.output.update(assets)
        return result

    @classmethod
    def from_data(
            cls,
            images: torch.Tensor,
            gt_image: torch.Tensor,
            net: torch.nn.Module,
            grid: Grid,
            submit: Submit,
            clip: int = 0,
    ):
        # prepare scales & flips
        scales = [cfg.default_scale]
        if cfg.multi_scale_inference:
            scales.extend(
                float(x)
                for x in cfg.model.extra_scales.split(',')
            )

        assert len(images.size()) == 4 and len(gt_image.size()) == 3
        assert images.size()[2:] == gt_image.size()[1:]
        input_size = images.size(2), images.size(3)
        if cfg.do_flip:
            flips = 1, 0
        else:
            flips = 0,

        # AMP: logits in fp16 to cut VRAM; accumulate in fp32 for stability
        use_amp = True

        with torch.inference_mode():
            output_accum = None

            for flip in flips:
                for scale in scales:
                    # flip/resize on GPU only as needed
                    x = cls.flip_tensor(images, 3) if flip == 1 else images
                    infer_size = [round(sz * scale) for sz in input_size]
                    if scale != 1.0:
                        x = cls.resize_tensor(x, infer_size)

                    inputs = dict(
                        images=x.cuda(non_blocking=True),
                        gts=gt_image.cuda(non_blocking=True)
                    )

                    with maybe_autocast(use_amp):
                        out = net(inputs)
                        pred = out["pred"]

                    # resize back to base scale if needed
                    if scale != 1.0:
                        pred = cls.resize_tensor(pred, input_size)

                    # flip back if needed
                    if flip == 1:
                        pred = cls.flip_tensor(pred, 3)

                    # upcast once to fp32 for accumulation; free per-iter refs ASAP
                    pred_f32 = pred.to(torch.float32)

                    # lazy allocate accumulator with correct shape/dtype/device
                    if output_accum is None:
                        output_accum = torch.zeros_like(pred_f32)

                    # accumulate and drop intermediates to release VRAM sooner
                    output_accum.add_(pred_f32)

                    # aggressively drop references to let the allocator reclaim
                    del pred_f32, pred, out


            probs = (
                output_accum
                .mul_(1.0 / (len(scales) * len(flips)))
                .softmax(dim=1),
            )
            del output_accum

            max_probs, preds = probs.max(1)
            del probs

            # predictions as uint8 on CPU (N,H,W)
            predictions: np.ndarray = (
                preds
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

            if cfg.segmentation.probability:
                # keep quantized prob on GPU; move later when writing
                prob_mask = (
                    max_probs
                    .mul_(255.0)
                    .clamp_(0, 255)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
            else:
                prob_mask = None

            # IoU on CPU
            _iou_acc = fast_hist(
                predictions.flatten(),
                gt_image.numpy().flatten(),
                cfg.DATASET.NUM_CLASSES,
            )


            result = cls(
                predictions=predictions,
                prob_mask=prob_mask,
                iou_acc=_iou_acc,
                grid=grid,
                error_mask=None,
                input_images=images,
                gt_images=gt_image,
                submit=submit,
                clip=clip,
            )
            return result



    @singledispatchmethod
    def clip_image(
            self,
            array,
    ):
        raise TypeError(f"Unsupported type for clip_image: {type(array)!r}")

    @clip_image.register
    def _clip_image_np(
            self,
            array: np.ndarray,
    ) -> np.ndarray:
        if self.clip <= 0:
            return array

        if array.ndim == 2:
            h, w = array.shape
            msg = (
                f"Invalid clip={self.clip} for array shape {array.shape}; "
                "slicing would produce an empty array"
            )
            assert h > 2 * self.clip and w > 2 * self.clip, msg
            view = array[self.clip:h - self.clip, self.clip:w - self.clip]
        elif array.ndim == 3:
            h, w, _ = array.shape
            msg = (
                f"Invalid clip={self.clip} for array shape {array.shape}; "
                "slicing would produce an empty array"
            )
            assert h > 2 * self.clip and w > 2 * self.clip, msg
            view = array[self.clip:h - self.clip, self.clip:w - self.clip, :]
        else:
            return array

        # ensure C-contiguous for cv2.imwrite, etc.
        result = np.ascontiguousarray(view)
        return result

    @clip_image.register
    def _clip_image_torch(
            self,
            array: torch.Tensor,
    ) -> torch.Tensor:
        if self.clip <= 0:
            return array

        if array.ndim == 2:
            h, w = array.shape
            msg = (
                f"Invalid clip={self.clip} for tensor shape {tuple(array.shape)}; "
                "slicing would produce an empty tensor"
            )
            assert h > 2 * self.clip and w > 2 * self.clip, msg
            view = array[self.clip:h - self.clip, self.clip:w - self.clip]
        elif array.ndim == 3:
            h, w = array.shape[-2], array.shape[-1]
            msg = (
                f"Invalid clip={self.clip} for tensor shape {tuple(array.shape)}; "
                "slicing would produce an empty tensor"
            )
            assert h > 2 * self.clip and w > 2 * self.clip, msg
            view = array[..., self.clip:h - self.clip, self.clip:w - self.clip]
        else:
            return array

        # pack for linear D2H copy or further GPU ops
        result = view.contiguous()
        return result

    @classmethod
    def flip_tensor(
            cls,
            x: torch.Tensor,
            dim: int
    ) -> torch.Tensor:
        """
        Flip Tensor along a dimension
        """
        if dim < 0:
            dim += x.dim()
        item = tuple(
            slice(None, None)
            if i != dim
            else torch.arange(
                x.size(i) - 1,
                -1,
                -1
            ).long()
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
            classid: int
    ) -> np.ndarray:
        """
        calculate class-specific error masks
        """
        # Class-specific error mask
        class_mask = gtruth >= 0
        class_mask &= gtruth == classid
        fp = pred == classid
        fp &= ~class_mask
        fp &= gtruth != cfg.DATASET.IGNORE_LABEL
        fn = pred != classid
        fn &= class_mask
        err_mask = fp | fn

        return err_mask.astype(int)

    @classmethod
    def calc_err_mask_all(
            cls,
            pred: np.ndarray,
            gtruth: np.ndarray,
    ) -> np.ndarray:
        """
        calculate class-agnostic error masks
        """
        result = gtruth >= 0
        result &= gtruth != cfg.DATASET.IGNORE_LABEL
        result &= pred != gtruth
        return result.astype(int)

    @cached_property
    def dump_percent(self) -> int:
        return 100

    def submit_colored(self):
        if self.predictions is None:
            return
        arrays = self.grid.colormap(to_numpy(self.predictions))
        files = self.grid.file.colored
        for array, file in zip(arrays, files):
            clipped = self.clip_image(array)
            self.submit.imwrite(file, clipped)

    def submit_probability(self):
        if self.prob_mask is None:
            return
        # arrays = (
        #     to_numpy(self.predictions)
        #     .__mul__(255)
        #     .astype(np.uint8)
        # )
        arrays = (
            self.prob_mask
            .cpu()
            .numpy()
        )
        files = self.grid.file.probability
        for array, file in zip(arrays, files):
            clipped = self.clip_image(array)
            self.submit.imwrite(file, clipped)


    def submit_output(self) -> None:
        if not self.output:
            return
        colorize = self.grid.colormap
        params = cv2.IMWRITE_PNG_COMPRESSION, 1
        for dirname, arrays in self.output.items():
            files = self.grid.file.output(dirname)
            # move only this tensor (no dict-wide to_numpy)
            arr = arrays
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().to('cpu')  # keep as tensor to avoid extra copies
                if 'pred_' in dirname:
                    # integer labels expected before colorize
                    arr = arr.argmax(dim=1).to(torch.uint8)
                    np_arr = arr.numpy()
                    np_arr = colorize(np_arr)  # colorize on CPU
                else:
                    np_arr = arr.numpy()
            else:
                np_arr = np.asarray(arr)

            for array, file in zip(np_arr, files):
                clipped = self.clip_image(array)
                self.submit.imwrite(file, clipped, params)

    def submit_grayscale(self):
        if self.predictions is None:
            grayscale = self.grid.file.grayscale.tolist()
            raise RuntimeError(f'No predictions to save for {grayscale}')
        arrays = self.predictions
        for array, file in zip(arrays, self.grid.file.grayscale):
            if array.ndim == 3:
                array = array.argmax(axis=-1)
            clipped = self.clip_image(array)
            self.submit.imwrite(file, clipped)

    def submit_all(self):
        # await previous writes
        submit = self.submit
        prev = self.submit.prev
        for future in prev:
            future.result()

        dt = time.time() - submit.t
        if dt > submit.period:
            # rotate thread pool
            msg = f'Rotating thread pool'
            logger.debug(msg)
            submit.threads.shutdown(wait=True)
            del submit.threads
            # trim memory
            try:
                msg = f'Trimming memory'
                logger.debug(msg)
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass

        next = submit.next
        if cfg.segmentation.probability:
            self.submit_probability()
        self.submit_output()
        self.submit_grayscale()
        if cfg.segmentation.colored:
            self.submit_colored()

        submit.prev = next
        del submit.next
        del submit.t
        _ = submit.t
