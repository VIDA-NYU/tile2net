from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import *
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

from tile2net.grid.cfg import cfg
from tile2net.tileseg.utils.misc import AverageMeter
from tile2net.tileseg.utils.misc import fast_hist, fmt_scale

if False:
    from .seggrid import SegGrid
    from tile2net.grid import Grid


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


@dataclass
class MiniBatch(

):
    input_images: torch.Tensor
    predictions: np.ndarray
    iou_acc: np.ndarray
    gt_images: torch.Tensor
    grid: SegGrid
    threads: ThreadPoolExecutor
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
            criterion: torch.nn.Module,
            val_loss: AverageMeter,
            grid: Grid,
            threads: ThreadPoolExecutor,
            clip:int =0
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

        calc_metrics = cfg.calc_metrics

        scales = [cfg.default_scale]
        if cfg.multi_scale_inference:
            it = (
                float(x)
                for x in cfg.extra_scales.split(',')
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
            threads=threads,
            clip=clip
        )
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

    def submit_all(
            self
    ) -> Iterator[Future]:
        if cfg.segment.probability:
            yield from self.submit_probability()
        # yield from self.submit_sidebyside()
        yield from self.submit_output()
        yield from self.submit_grayscale()
        if cfg.segment.colored:
            yield from self.submit_colored()

    def submit_probability(self):
        if self.prob_mask is None:
            return
        arrays = (
            to_numpy(self.prob_mask)
            .__mul__(255)
            .astype(np.uint8)
        )
        files = self.grid.file.probability
        for array, file in zip(arrays, files):
            # Clip the image before saving
            clipped_array = self.clip_image(array)
            future = self.threads.submit(cv2.imwrite, file, clipped_array)
            yield future

    def submit_error(self):
        if self.error_mask is None:
            return
        # todo @mary-h86 see this func. It doesn't seem to be saving err masks
        #   it just saves the prediction, will return to this later
        raise NotImplementedError

    def submit_sidebyside(self):
        it = zip(cfg.DATASET.MEAN, cfg.DATASET.STD)
        inv_mean = [-mean / std for mean, std in it]
        inv_std = [1 / std for std in cfg.DATASET.STD]
        if not cfg.dump_percent:
            return
        INPUT_IMAGE = to_numpy(
            standard_transforms
            .Normalize(mean=inv_mean, std=inv_std)
            (self.input_images)
        )
        files = self.grid.file.sidebyside
        it = zip(INPUT_IMAGE, self.predictions, files)
        for input_image, prediction, file in it:
            self.dump_percent += cfg.dump_percent
            if self.dump_percent < 100:
                continue
                
            # Apply clipping to the input image array before converting to PIL
            if self.clip > 0:
                input_image = self.clip_image(input_image)
                # Also clip the prediction array
                prediction = self.clip_image(prediction)
                
            input_image = (
                standard_transforms.ToPILImage()
                (input_image)
                .convert('RGB')
            )
            prediction_pil = cfg.DATASET_INST.colorize_mask(prediction)

            size = input_image.width * 2, input_image.height
            composited = Image.new('RGB', size)
            composited.paste(input_image, (0, 0))
            composited.paste(prediction_pil, (prediction_pil.width, 0))
            future = self.threads.submit(composited.save, file)
            yield future

    def submit_output(self):
        colorize = self.grid.colormap
        it = to_numpy(self.output).items()
        for dirname, arrays in it:
            files = self.grid.file.output(dirname)
            if 'pred_' in dirname:
                arrays = colorize(arrays)
            for array, file in zip(arrays, files):
                # Clip the image before saving
                clipped_array = self.clip_image(array)
                future = self.threads.submit(cv2.imwrite, file, clipped_array)
                yield future

    def submit_grayscale(self):
        if self.predictions is None:
            return
        arrays = to_numpy(self.predictions)
        for array, file in zip(arrays, self.grid.file.grayscale):

            if array.ndim == 3:  # one-hot or logits → class map2
                array = array.argmax(axis=-1)
            
            # Clip the image before saving
            clipped_array = self.clip_image(array)
            future = self.threads.submit(
                cv2.imwrite,
                file,
                clipped_array.astype('uint8')
            )
            yield future

    def clip_image(self, array: np.ndarray) -> np.ndarray:
        """
        Clip the image by removing the specified number of pixels from each edge.
        
        Args:
            array: The image array to clip
            
        Returns:
            The clipped image array
        """
        if self.clip <= 0:
            return array
            
        # Get the dimensions of the array
        if array.ndim == 2:
            h, w = array.shape
            return array[self.clip:h-self.clip, self.clip:w-self.clip]
        elif array.ndim == 3:
            h, w, c = array.shape
            return array[self.clip:h-self.clip, self.clip:w-self.clip, :]
        else:
            # If the array has an unexpected number of dimensions, return it unchanged
            return array
            
    def submit_colored(self):
        """
        Colorized segmentation mask where different classes (road, sidewalk, crosswalk) are represented by different colors according to a predefined color palette
        """
        if self.predictions is None:
            return
        arrays = to_numpy(self.predictions)
        arrays = self.grid.colormap(arrays)
        files = self.grid.file.colored
        for array, file in zip(arrays, files):
            # Clip the image before saving
            clipped_array = self.clip_image(array)
            future = self.threads.submit(cv2.imwrite, file, clipped_array)
            yield future
