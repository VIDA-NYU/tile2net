from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import *
from typing import Optional

import cv2
import geopandas as gpd
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

import tile2net.tileseg.datasets.base_loader
import tile2net.tileseg.inference.inference
import tile2net.tileseg.utils.misc
import tile2net.tileseg.utils.trnval_utils
from tile2net.tiles.cfg import cfg
from tile2net.tiles.util import look_at
from tile2net.tileseg.utils.misc import AverageMeter
from tile2net.tileseg.utils.misc import fast_hist, fmt_scale
from .dir import Dir
from .mask2poly import Mask2Poly

if False:
    from tile2net.tiles import Tiles


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
    tiles: Tiles
    output: dict[str, torch.Tensor] = field(default_factory=dict)
    prob_mask: Optional[torch.Tensor] = None
    error_mask: Optional[np.ndarray] = None

    @classmethod
    @look_at(tile2net.tileseg.utils.trnval_utils.eval_minibatch)
    def from_data(
            cls,
            images,
            gt_image,
            net: torch.nn.Module,
            criterion: torch.nn.Module,
            val_loss: AverageMeter,
            tiles: Tiles,
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
            tiles=tiles,
            error_mask=err_mask,
            input_images=images,
            gt_images=gt_image,
        )
        return result

    @classmethod
    @look_at(tile2net.tileseg.utils.trnval_utils.flip_tensor)
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
    @look_at(tile2net.tileseg.utils.trnval_utils.resize_tensor)
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
    @look_at(tile2net.tileseg.utils.trnval_utils.calc_err_mask)
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
    @look_at(tile2net.tileseg.utils.trnval_utils.calc_err_mask_all)
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
    def threads(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor()

    @cached_property
    def futures(self) -> list[Future]:
        return []

    @cached_property
    def dump_percent(self) -> int:
        return 100

    def await_all(self) -> Self:
        wait(self.futures)
        for fut in self.futures:
            fut.result()
        self.futures.clear()
        return self

    def submit_all(self) -> Self:
        self.submit_probability()
        self.submit_error()
        self.submit_sidebyside()
        self.submit_output()
        self.submit_raw()
        self.submit_mask()
        self.submit_polygons()
        return self

    @look_at(tile2net.tileseg.utils.misc.ThreadedDumper.save_prob_and_err_mask)
    @look_at(Dir.iterator)
    def submit_probability(self):
        if self.prob_mask is None:
            return
        arrays = (
            to_numpy(self.prob_mask)
            .__mul__(255)
            .astype(np.uint8)
        )
        files = next(self.tiles.outdir.seg_results.prob.iterator())
        # len(list(self.tiles.outdir.seg_results.prob.iterator()))
        # len(self.tiles.outdir.seg_results.files())
        # len(arrays)
        for array, file in zip(arrays, files):
            future = self.threads.submit(cv2.imwrite, file, array)
            self.futures.append(future)

    @look_at(tile2net.tileseg.utils.misc.ThreadedDumper.save_prob_and_err_mask)
    def submit_error(self):
        if self.error_mask is None:
            return
        # todo @mary-h86 see this func. It doesn't seem to be saving err masks
        #   it just saves the prediction, will return to this later
        look_at(tile2net.tileseg.utils.misc.ThreadedDumper.save_prob_and_err_mask)
        raise NotImplementedError

    @look_at(tile2net.tileseg.utils.misc.ThreadedDumper.create_composite_image)
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
        files = next(self.tiles.outdir.seg_results.sidebyside.iterator())
        it = zip(INPUT_IMAGE, self.predictions, files)
        for input_image, prediction, file in it:
            self.dump_percent += cfg.dump_percent
            if self.dump_percent < 100:
                continue
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
            self.futures.append(future)

    @look_at(tile2net.tileseg.utils.misc.ThreadedDumper.get_dump_assets)
    def submit_output(self):
        colorize = self.tiles.colormap
        it = to_numpy(self.output).items()
        for dirname, arrays in it:
            files = next(self.tiles.outdir.outputs.iterator(dirname))
            if 'pred_' in dirname:
                arrays = colorize(arrays)
            for array, file in zip(arrays, files):
                future = self.threads.submit(cv2.imwrite, file, array)
                self.futures.append(future)

    @look_at(tile2net.tileseg.datasets.base_loader.BaseLoader.dump_images)
    def submit_raw(self):
        """
        Raw segmentation mask without colorization, containing class IDs as pixel values.
        """
        if self.predictions is None:
            return
        arrays = to_numpy(self.predictions)
        files = next(self.tiles.outdir.raw.iterator())
        for array, file in zip(arrays, files):
            future = self.threads.submit(cv2.imwrite, file, array)
            self.futures.append(future)

    @look_at(tile2net.tileseg.datasets.base_loader.BaseLoader.dump_images)
    def submit_mask(self):
        """
        Colorized segmentation mask where different classes (road, sidewalk, crosswalk) are represented by different colors according to a predefined color palette
        """
        if self.predictions is None:
            return
        arrays = to_numpy(self.predictions)
        arrays = self.tiles.colormap(arrays)
        files = next(self.tiles.outdir.mask.iterator())
        for array, file in zip(arrays, files):
            future = self.threads.submit(cv2.imwrite, file, array)
            self.futures.append(future)

    @look_at(tile2net.tileseg.inference.inference.Inference.validate)
    def submit_polygons(self):
        affines = next(self.tiles.predtiles.affine_iterator())
        arrays = to_numpy(self.predictions).astype(np.uint8)
        files = next(self.tiles.outdir.polygons.iterator())
        it = zip(arrays, affines, files)
        for array, affine, file in it:
            frame = (
                Mask2Poly
                .from_array(array=array, affine=affine)
                .pipe(gpd.GeoDataFrame)
            )
            future = self.threads.submit(frame.to_parquet, file)
            self.futures.append(future)
