from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch, cached_property
from typing import *
from typing import Optional

import cv2
import numpy as np
import torch
from torch.nn.parallel import DataParallel

from tile2net.grid.cfg import cfg
from tile2net.tileseg.utils.misc import fast_hist
from .submit import Submit

# cv2.setNumThreads(1)

if TYPE_CHECKING:
    from .seggrid import SegGrid


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

        msg = f"Invalid clip={clip} for shape {array.shape}"
        assert (
                h > 2 * clip
                and w > 2 * clip
        ), msg
        view = array[clip:-clip, clip:-clip]

    elif nd == 3:
        # assume channel-last (H, W, C)
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
            scales.extend(cfg.model.extra_scales)
            scales.sort()

        input_size = images.size(2), images.size(3)
        if cfg.do_flip:
            flips = 1, 0
        else:
            flips = 0,

        has_meaningful_gt = not cls._is_placeholder_gt(gt_image)

        with torch.inference_mode():
            # preallocate accumulator to avoid repeated allocations
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

                    if torch.cuda.is_available():
                        device_type = 'cuda'
                    else:
                        device_type = 'cpu'
                    with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                        out = net(inputs)
                        pred = out["pred"]

                    # resize back to base scale if needed
                    if scale != 1.0:
                        pred = cls.resize_tensor(pred, input_size)

                    # flip back if needed
                    if flip == 1:
                        pred = cls.flip_tensor(pred, 3)

                    pred_f32 = pred.to(torch.float32)

                    # if first iteration, allocate accum to be like pred
                    if output_accum is None:
                        output_accum = torch.zeros_like(pred_f32)
                    output_accum.add_(pred_f32)

                    # aggressively drop refs
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

    def submit_prob(self) -> None:
        """Submit probability maps for parallel writing to disk.

        Uses CUDA event synchronization to ensure GPU->CPU transfer completes
        before handing off to thread pool for parallel compression and writing.
        """
        probs = (
            self.probs
            .to(torch.float16)
            # avoids sequential sync
            .to('cpu', non_blocking=True)
        )
        stream = torch.cuda.current_stream()
        event = stream.record_event()
        files = list(self.seggrid.file.prob)

        def write_batch():
            # wait once for all gpu->cpu transfers to complete
            event.synchronize()

            it = zip(files, probs.numpy())
            futures = [
                self.submit
                .to_tiff(file, prob)
                for file, prob in it
            ]

            for future in futures:
                future.result()

        # thread-within-thread necessary bc event.synchronize() is blocking
        future = self.submit.threads.submit(write_batch)
        self.submit.next.append(future)

    def submit_pred(self):
        """Submit prediction masks for parallel writing to disk.

        Uses async GPU->CPU transfer and batched submission to avoid
        blocking the GPU pipeline during file I/O operations.
        """
        pred_tensor = (
            self.max.pred
            .to(torch.uint8)
            .to('cpu', non_blocking=True)
        )
        event = (
            torch.cuda
            .current_stream()
            .record_event()
        )
        files = list(self.seggrid.file.pred)

        def write_batch():
            # wait once for gpu->cpu transfer to complete
            event.synchronize()

            it = zip(files, pred_tensor.numpy())
            args = [cv2.IMWRITE_PNG_COMPRESSION, 1]

            futures = [
                self.submit
                .imwrite(file, prob, args)
                for file, prob in it
            ]

            for future in futures:
                future.result()

        # thread-within-thread necessary bc event.synchronize() is blocking
        future = self.submit.threads.submit(write_batch)
        self.submit.next.append(future)
