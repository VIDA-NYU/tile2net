from __future__ import annotations
from torch.nn.parallel import DataParallel
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
import pandas as pd
import torch

from tile2net.grid.cfg import cfg
from tile2net.grid.cfg.logger import logger
from tile2net.tileseg.utils.misc import fast_hist, fmt_scale
from .submit import Submit

# cv2.setNumThreads(1)

if False:
    from .seggrid import SegGrid
    from tile2net.grid import Grid
    from tile2net.grid.ingrid import InGrid


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
class Max:
    prob: torch.Tensor
    pred: torch.Tensor

    @classmethod
    def from_probs(
            cls,
            probs: torch.Tensor,
            dim=1,
    ) -> Self:
        """"""
        pred: torch.Tensor
        prob: torch.Tensor
        prob, pred = probs.max(dim=dim)
        pred = pred.to(torch.uint8)
        result = cls(prob=prob, pred=pred)
        return result

    @cached_property
    def quantized(self) -> torch.Tensor:
        result = (
            self.prob
            .mul(255.0)
            .clamp_(0, 255)
            .to(torch.uint8)
        )
        return result

    @cached_property
    def colors(self) -> torch.Tensor:
        return cfg.colormap(self.pred)


class Foreground(Max):
    @classmethod
    def from_probs(
            cls,
            probs: torch.Tensor,
            dim=1,
    ) -> Self:
        keep = torch.arange(probs.size(1), device=probs.device)
        x = torch.arange(probs.shape[1])[cfg.dataset.ignore_label]
        keep = keep[keep != x]
        fore = torch.index_select(probs, 1, keep)
        return super().from_probs(fore, dim)

    @cached_property
    def colors(self):
        colors = super(Foreground, self).colors
        result = (
            colors
            .to(torch.float32)
            .mul_(self.prob.unsqueeze(-1))
            .round_()
            .clamp_(0., 255.)
            .to(torch.uint8)
        )
        return result


@dataclass
class MiniBatch:
    probs: Optional[torch.Tensor]
    seggrid: SegGrid
    submit: Submit

    @classmethod
    def _is_placeholder_gt(cls, gt_image: torch.Tensor) -> bool:
        """
        Check if ground truth tensor contains only placeholder values (-1).
        Returns True if all values are -1, indicating no meaningful ground truth.
        """
        out = (
            gt_image
            .eq(-1)
            .all()
            .item()
        )
        return out

    @classmethod
    def from_data(
            cls,
            images: torch.Tensor,
            gt_image: torch.Tensor,
            net: DataParallel | torch.nn.Module,
            seggrid: SegGrid,
            submit: Submit,
            clip: int = 0,
    ):

        # prepare scales & flips
        scales = [cfg.default_scale]
        if cfg.multi_scale_inference:
            scales.extend( cfg.model.extra_scales )
            scales.sort()

        input_size = images.size(2), images.size(3)
        if cfg.do_flip:
            flips = 1, 0
        else:
            flips = 0,

        # AMP: logits in fp16 to cut VRAM; accumulate in fp32 for stability
        use_amp = True

        # Check if ground truth is meaningful or just placeholder
        has_meaningful_gt = not cls._is_placeholder_gt(gt_image)

        with torch.inference_mode():
            output_accum = None

            for flip in flips:
                for scale in scales:
                    # flip/resize on GPU only as needed
                    x = cls.flip_tensor(images, 3) if flip == 1 else images
                    infer_size = [round(sz * scale) for sz in input_size]
                    if scale != 1.0:
                        x = cls.resize_tensor(x, infer_size)

                    inputs = dict(images=x.cuda(non_blocking=True))
                    if has_meaningful_gt:
                        inputs['gts'] = gt_image.cuda(non_blocking=True)


                    assert cfg.dataset.num_classes == 4
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

            probs = clip_image(
                output_accum
                .div_(len(scales) * len(flips))
                .softmax(dim=1),
                clip
            )
            result = cls(
                probs=probs,
                seggrid=seggrid,
                submit=submit,
            )
            return result

    @cached_property
    def max(self) -> Max:
        """Constructs Max data"""
        return Max.from_probs(self.probs)

    @cached_property
    def foreground(self) -> Foreground:
        return Foreground.from_probs(self.probs)


    @cached_property
    def intensity(self) -> torch.Tensor:
        maxc = self.max.colors
        fore = self.foreground.colors

        black_pixel_mask = (
            maxc
            .sum(dim=-1, keepdim=True)
            .eq(0)
        )
        result = torch.where(black_pixel_mask, fore, maxc)

        return result

    @cached_property
    def iou_acc(self):

        raise NotImplementedError
        return fast_hist(
            self.preds.flatten(),
            self.gt_images.numpy().flatten(),
            cfg.DATASET.NUM_CLASSES,
        )

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
            )
            .long()
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

    def imwrite(
            self,
            tensor: torch.Tensor,
            files: pd.Series
    ) -> None:
        arrays = (
            tensor
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        for array, file in zip(arrays, files):
            self.submit.imwrite(file, array)

    def submit_colorized(self):
        self.imwrite(self.max.colors, self.seggrid.file.colorized)

    def submit_intensity(self):
        self.imwrite(self.intensity, self.seggrid.file.intensity)

    # def submit_prob(self) -> None:
    #     probs = (
    #         self.probs
    #         .to(torch.float16)
    #         .cpu()
    #         .numpy()
    #     )
    #     for array, file in zip(probs, self.seggrid.file.prob):
    #         self.submit.to_tiff(file, array)


    def write_all(self, probs_tensor, event, files):
        event.synchronize()
        probs_np = probs_tensor.numpy()
        for arr, f in zip(probs_np, files):
            self.submit._to_tiff(f, arr)

    def submit_prob(self) -> None:
        probs = self.probs.to(torch.float16).to('cpu', non_blocking=True)
        stream = torch.cuda.current_stream()
        event = stream.record_event()
        files = list(self.seggrid.file.prob)
        future = self.submit.threads.submit(self.write_all, probs, event, files)
        self.submit.next.append(future)


    # def submit_output(self) -> None:
    #     if not self.output:
    #         return
    #     colorize = self.grid.colormap
    #     for dirname, arrays in self.output.items():
    #         files = self.grid.file.output(dirname)
    #         # move only this tensor (no dict-wide to_numpy)
    #         arr = arrays
    #         if isinstance(arr, torch.Tensor):
    #             arr = arr.detach().to('cpu')  # keep as tensor to avoid extra copies
    #             if 'pred_' in dirname:
    #                 # integer labels expected before colorize
    #                 arr = arr.argmax(dim=1).to(torch.uint8)
    #                 np_arr = arr.numpy()
    #                 np_arr = colorize(np_arr)  # colorize on CPU
    #             else:
    #                 np_arr = arr.numpy()
    #         else:
    #             np_arr = np.asarray(arr)
    #
    #         for array, file in zip(np_arr, files):
    #             self.submit.imwrite(file, array)

    def submit_pred(self):
        arrays = (
            self.max.pred
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        for array, file in zip(arrays, self.seggrid.file.pred):
            if array.ndim == 3:
                array = array.argmax(axis=-1)
            self.submit.imwrite(file, array)

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
        # if cfg.segmentation.probability:
        #     self.submit_probability()
        # self.submit_output()
        self.submit_pred()
        if cfg.segmentation.colorized:
            self.submit_colorized()
        if cfg.segmentation.intensity:
            self.submit_intensity()

        submit.prev = next
        del submit.next
        del submit.t
        _ = submit.t

    # @classmethod
    # def set_columns(
    #         cls,
    #         ingrid: InGrid
    # ):
    #     """
    #     Temporary workaround to have the columns show up in the SegGrid
    #     before the subprocess
    #     """
    #     seggrid = ingrid.seggrid.broadcast
    #     predict = seggrid.predict
    #     _ = (
    #         seggrid.file.colorized,
    #         seggrid.file.intensity,
    #         seggrid.file.pred,
    #         seggrid.file.prob,
    #     )
    #     seggrid.predict = predict
