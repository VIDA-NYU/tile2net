from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import singledispatch, cached_property
from pathlib import Path
from typing import *

import numpy as np
import tifffile
import torch
from torch.nn.parallel import DataParallel

from tile2net.xyz.cfg import cfg
from tile2net.tileseg.utils.misc import fast_hist
from .submit import Submit

if TYPE_CHECKING:
    from .predict import Predict


def to_tiff(
        filename: str,
        arr: np.ndarray,
) -> None:
    p = Path(filename)
    parent = p.parent
    parent.mkdir(parents=True, exist_ok=True)
    tid = threading.get_ident()
    tmp = parent / f'tmp.{tid}.{p.name}'
    tmp_str = str(tmp)
    p_str = str(p)

    try:
        tifffile.imwrite(tmp_str, arr, compression='zlib')
        os.replace(tmp_str, p_str)
    except Exception:
        try:
            if os.path.exists(tmp_str):
                os.unlink(tmp_str)
        finally:
            raise


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

        msg = f"Invalid clip={clip} for shape {array.shape}"
        assert (
                h > 2 * clip
                and w > 2 * clip
        ), msg
        view = array[clip:-clip, clip:-clip]

    elif nd == 3:
        # detect channel-first vs channel-last based on which dimension is smallest
        # probability maps are (K, H, W) where K is typically small (< 10)
        # images would be (H, W, C) where C is typically 3 or 4
        dim_sizes = array.shape
        smallest_dim = np.argmin(dim_sizes)

        if smallest_dim == 0:
            # channel-first (K, H, W)
            h, w = array.shape[1], array.shape[2]
            msg = f"Invalid clip={clip} for shape {array.shape}"
            assert (
                    h > 2 * clip
                    and w > 2 * clip
            ), msg
            view = array[:, clip:-clip, clip:-clip]
        else:
            # channel-last (H, W, C)
            h, w = array.shape[0], array.shape[1]
            msg = f"Invalid clip={clip} for shape {array.shape}"
            assert (
                    h > 2 * clip
                    and w > 2 * clip
            ), msg
            view = array[clip:-clip, clip:-clip, :]

    elif nd == 4:
        # support both (N, C, H, W) and (N, H, W, C)
        # decide by which axes look like spatial
        n0, n1, n2, n3 = array.shape
        # # if channel-first -> (N,C,H,W): spatial = (-2,-1)
        if (
                n2 >= 2 * clip + 1
                and n3 >= 2 * clip + 1
        ):
            # assume channel-first (N,C,H,W)
            view = array[:, :, clip:-clip, clip:-clip]
        elif (
                n1 >= 2 * clip + 1
                and n2 >= 2 * clip + 1
        ):
            view = array[:, clip:-clip, clip:-clip, :]
        else:
            raise AssertionError(f"Invalid clip={clip} for shape {array.shape}")

    else:
        # leave uncommon ranks unchanged
        return array

    out = np.ascontiguousarray(view)
    return out


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
        msg = f"Invalid clip={clip} for shape {tuple(array.shape)}"
        assert (
                h > 2 * clip
                and w > 2 * clip
        ), msg
        view = array[clip:-clip, clip:-clip]

    elif nd == 3:
        # treat as (..., H, W) if it’s actually CHW or NHW?
        # be explicit: for 3D tensors we assume channel-first (C,H,W)
        h, w = array.shape[-2], array.shape[-1]
        msg = f"Invalid clip={clip} for shape {tuple(array.shape)}"
        assert (
                h > 2 * clip
                and w > 2 * clip
        ), msg
        view = array[..., clip:-clip, clip:-clip]

    elif nd == 4:
        # common case: (N, C, H, W) -> clip last two dims
        h, w = array.shape[-2], array.shape[-1]
        msg = f"Invalid clip={clip} for shape {tuple(array.shape)}"
        assert (
                h > 2 * clip
                and w > 2 * clip
        ), msg
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
    probs: torch.Tensor
    """Clipped predicted probability scores for the minibatch."""
    unclipped_probs: torch.Tensor | None
    """Unclipped predicted probability scores for the minibatch."""
    submit: Submit
    """Submit object for coordinating and parallelizing file I/O."""
    pred_paths: list[str]
    """File paths to save predicted segmentation masks."""
    prob_paths: list[str]
    """File paths to save predicted probability scores."""
    unclipped_prob_paths: list[str] | None
    """File paths to save unclipped predicted probability scores."""

    @classmethod
    def from_data(
            cls,
            images: torch.Tensor,
            masks: torch.Tensor,
            net: DataParallel | torch.nn.Module,
            pred_paths: list[str],
            prob_paths: list[str],
            submit: Submit,
            clip: int = 0,
            unclipped_prob_paths: list[str] | None = None,
    ):
        """
        Perform inference on a minibatch of images.

        See its use in the prediction module:
            >>> Predict.__iter__

        Returns:
            A MiniBatch dataclass instance, providing the
            probabilities and filepaths necessary for serialization.
        """
        scales = [cfg.default_scale]
        if cfg.multi_scale_inference:
            scales.extend(cfg.model.extra_scales)
            scales.sort()

        input_size = images.size(2), images.size(3)
        if cfg.do_flip:
            flips = True, False
        else:
            flips = False,

        has_meaningful_gt = not cls._is_placeholder_gt(masks)
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'

        with torch.inference_mode():
            # preallocate accumulator to avoid repeated allocations
            output_accum = None

            for flip in flips:
                for scale in scales:
                    # flip/resize on GPU only as needed
                    if flip:
                        x = cls.flip_tensor(images, 3)
                    else:
                        x = images

                    infer_size = [
                        round(sz * scale)
                        for sz in input_size
                    ]
                    if scale != 1.0:
                        x = cls.resize_tensor(x, infer_size)

                    inputs = dict(images=x.cuda(non_blocking=True))
                    if has_meaningful_gt:
                        inputs['gts'] = masks.cuda(non_blocking=True)

                    with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                        out = net(inputs)
                        pred = out["pred"]

                    # resize back to base scale if needed
                    if scale != 1.0:
                        pred = cls.resize_tensor(pred, input_size)

                    # flip back if needed
                    if flip:
                        pred = cls.flip_tensor(pred, 3)

                    pred_f32 = pred.to(torch.float32)

                    # if first iteration, allocate accum to be like pred
                    if output_accum is None:
                        output_accum = torch.zeros_like(pred_f32)
                    output_accum.add_(pred_f32)

                    # aggressively drop refs
                    del pred_f32, pred, out

            averaged = (
                output_accum
                .div_(len(scales) * len(flips))
                .softmax(dim=1)
            )

            if unclipped_prob_paths is not None:
                unclipped_probs = averaged.clone()
                probs = clip_image(averaged, clip)
            else:
                unclipped_probs = None
                probs = clip_image(averaged, clip)

            result = cls(
                probs=probs,
                unclipped_probs=unclipped_probs,
                pred_paths=pred_paths,
                prob_paths=prob_paths,
                unclipped_prob_paths=unclipped_prob_paths,
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
        # class-specific error mask
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

    def submit_prob(self):
        """Submit probability maps for parallel writing to disk.

        Uses CUDA event synchronization to ensure GPU->CPU transfer completes
        before dispatching parallel file writes to the I/O pool.
        """
        probs = (
            self.probs
            .to(torch.float16)
            .to('cpu', non_blocking=True)
        )
        event = (
            torch.cuda
            .current_stream()
            .record_event()
        )
        submit = self.submit

        def coordinate():
            # wait for gpu->cpu transfer to complete
            event.synchronize()
            it = zip(self.prob_paths, probs.numpy())

            futures = [
                submit.submit_io(to_tiff, file, prob)
                for file, prob in it
            ]
            for future in futures:
                future.result()

        submit.submit_batch(coordinate)

    def submit_pred(self):
        """Submit prediction masks for parallel writing to disk.

        Uses async GPU->CPU transfer and batched submission to avoid
        blocking the GPU pipeline during file I/O operations.
        """
        preds = (
            self.max.pred
            .to(torch.uint8)
            .to('cpu', non_blocking=True)
        )
        event = (
            torch.cuda
            .current_stream()
            .record_event()
        )
        submit = self.submit

        def coordinate():
            # wait for gpu->cpu transfer to complete
            event.synchronize()
            it = zip(self.pred_paths, preds.numpy())
            futures = [
                submit.submit_io(to_tiff, file, pred)
                for file, pred in it
            ]
            for future in futures:
                future.result()

        submit.submit_batch(coordinate)

    def submit_unclipped_prob(self):
        """Submit unclipped probability maps for parallel writing to disk.

        Uses CUDA event synchronization to ensure GPU->CPU transfer completes
        before dispatching parallel file writes to the I/O pool.
        """
        if self.unclipped_probs is None or self.unclipped_prob_paths is None:
            return

        unclipped_probs = (
            self.unclipped_probs
            .to(torch.float16)
            .to('cpu', non_blocking=True)
        )
        event = (
            torch.cuda
            .current_stream()
            .record_event()
        )
        submit = self.submit

        def coordinate():
            # wait for gpu->cpu transfer to complete
            event.synchronize()
            it = zip(self.unclipped_prob_paths, unclipped_probs.numpy())

            futures = [
                submit.submit_io(to_tiff, file, prob)
                for file, prob in it
            ]
            for future in futures:
                future.result()

        submit.submit_batch(coordinate)

    def __len__(self):
        return self.probs.size(0)

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
