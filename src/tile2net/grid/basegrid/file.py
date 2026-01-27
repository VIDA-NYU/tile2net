from __future__ import annotations

import os
import tifffile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from tile2net.grid.frame.namespace import namespace
from ..cfg import cfg

if False:
    from tile2net.grid.frame import column
    from tile2net.grid.basegrid.basegrid import BaseGrid
    from tile2net.grid.grid import Grid

from .. import frame


class File(
    namespace
):
    instance: BaseGrid
    @frame.column
    def static(self):
        ...

    @property
    def basegrid(self) -> BaseGrid:
        return self.instance

    @frame.column
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.file.pred
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        msg = f'Must be overriden in a subclass of {File.__name__}'
        raise NotImplementedError(msg)
        # todo: tell user to look at specific grid.file.pred

    @frame.column
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.file.prob
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        msg = f'Must be overriden in a subclass of {File.__name__}'
        raise NotImplementedError(msg)
        # todo: tell user to look at specific grid.file.prob

    @frame.column
    def colorized(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.

        Lazily generated from prob and pred files already saved to disk.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.file.Colorized
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.basegrid
        name = grid.__name__
        FILES = (
            getattr(grid.outdir, name)
            .colorized.files(grid)
        )
        if self:
            return FILES
        self.colorized = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            probs = self.prob[loc]
            preds = self.pred[loc]
            files = FILES[loc]
            max_workers = min(self.basegrid.cfg.compress_workers, len(files))
            it = zip(probs, preds, files)
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                futures: dict[Future, str] = {
                    threads.submit(self._compute_colorized, prob, pred, file):
                        file
                    for prob, pred, file in it
                }
                for future, file in futures:
                    future.result()

            assert (
                files
                .map(os.path.exists)
                .all()
            )

        return FILES

    @frame.column
    def intensity(self) -> pd.Series:
        """
        File-paths to intensity representation of segmentation results.

        Lazily generated from prob and pred files already saved to disk.
        Alternative visualization format where segmentation classes are represented
        as intensity values.

        Returns:
            pd.Series: File paths to intensity representations for each seg-tile

        Example:
            >>> grid: Grid
            >>> grid.seggrid.file.intensity
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.basegrid
        name = grid.__name__
        FILES = (
            getattr(grid.grid.outdir, name)
            .intensity.files(grid)
        )
        if self:
            return FILES
        self.intensity = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            files = FILES.loc[loc]
            probs = self.prob.loc[loc]
            preds = self.pred.loc[loc]

            max_workers = min(len(files), self.basegrid.cfg.compress_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                it = zip(probs, preds, files)
                futures: dict[Future, str] = {
                    threads.submit(self._compute_intensity, prob, pred, file):
                        file
                    for prob, pred, file in it
                }
                for future, file in futures:
                    future.result()

            assert (
                files
                .map(os.path.exists)
                .all()
            )

        return FILES

    @frame.column
    def sidebyside(self) -> pd.Series:
        """
        File-paths to side-by-side composite images of input and prediction.

        Lazily generated from static and colorized files already saved to disk.
        Creates composite images with the original input on the left and the
        colorized prediction on the right.

        Returns:
            pd.Series: File paths to side-by-side composites for each seg-tile

        Example:
            >>> grid: Grid
            >>> grid.seggrid.file.sidebyside
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.basegrid
        name = grid.__name__
        FILES = (
            getattr(grid.outdir, name)
            .sidebyside.files(grid)
        )
        if self:
            return FILES
        self.sidebyside = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            files = FILES.loc[loc]
            statics = self.static.loc[loc]
            colorized = self.colorized.loc[loc]

            max_workers = min(len(files), self.basegrid.cfg.compress_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                it = zip(statics, colorized, files)
                futures: dict[Future, str] = {
                    threads.submit(self._compute_sidebyside, static, colorized_file, file):
                        file
                    for static, colorized_file, file in it
                }
                for future, file in futures:
                    future.result()

            assert (
                files
                .map(os.path.exists)
                .all()
            )

        return FILES

    @frame.column
    def overlay(self) -> pd.Series:
        """
        File-paths to overlay images with colorized predictions blended onto input.

        Lazily generated from static and colored files already saved to disk.
        Creates overlay images with 20% opacity colorized predictions on top of
        original input, with 0% opacity for black pixels from the colormaps.

        Returns:
            pd.Series: File paths to overlay images for each seg-tile

        Example:
            >>> grid: Grid
            >>> grid.seggrid.file.overlay
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.basegrid
        name = grid.__name__
        FILES = (
            getattr(grid.grid.outdir, name)
            .overlay.files(grid)
        )
        if self:
            return FILES
        self.overlay = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            files = FILES.loc[loc]
            statics = self.static.loc[loc]
            colored = self.colored.loc[loc]

            max_workers = min(len(files), self.basegrid.cfg.compress_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                it = zip(statics, colored, files)
                futures: dict[Future, str] = {
                    threads.submit(self._compute_overlay, static, colored_file, file):
                        file
                    for static, colored_file, file in it
                }
                for future, file in futures:
                    future.result()

            assert (
                files
                .map(os.path.exists)
                .all()
            )

        return FILES

    @frame.column
    def error(self) -> pd.Series:
        """
        File-paths to error masks comparing predictions to ground truth.

        Lazily generated from prob and pred files already saved to disk.

        Example:
            >>> grid: Grid
            >>> grid.seggrid.file.error
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        raise NotImplementedError('requires GT')
        grid = self.obj
        name = grid.__name__
        FILES = (
            getattr(grid.grid.outdir, name)
            .error.files(grid)
        )
        self.error = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            probs = self.prob[loc]
            preds = self.pred[loc]
            files = files[loc]

            max_workers = min(self.obj.cfg.compress_workers, len(files))
            it = zip(probs, preds, files)
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                ...

            assert (
                FILES
                .map(os.path.exists)
                .all()
            )

        return FILES

    @frame.column
    def soft(self) -> pd.Series:
        """
        File-paths to soft segmentation visualizations (probability-weighted colors).

        Lazily generated from prob files.
        Visualizes uncertainty by blending class colors based on their predicted probabilities.
        A pixel with 50% Class A (Red) and 50% Class B (Green) appears Yellow.
        A pixel with 80% Background (Black) and 20% Class A (Red) appears Faint Red.

        Returns:
            pd.Series: File paths to soft segmentation images.

        # todo: examples
        """
        grid = self.basegrid
        name = grid.__name__
        FILES = (
            getattr(grid.grid.outdir, name)
            .soft.files(grid)
        )
        if self:
            return FILES
        self.soft = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            files = FILES.loc[loc]
            probs = self.prob.loc[loc]

            max_workers = min(len(files), self.basegrid.cfg.compress_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                futures: dict[Future, str] = {
                    threads.submit(self._compute_soft, prob, file): file
                    for prob, file in zip(probs, files)
                }
                for future, file in futures:
                    future.result()

            assert (
                files
                .map(os.path.exists)
                .all()
            )

        return FILES

    @classmethod
    def _compute_colorized(
            cls,
            pred_file: str,
            output_file: str,
    ) -> str:
        """Generate colorized visualization from prob and pred files."""
        pred = tifffile.imread(pred_file)
        pred = torch.from_numpy(pred)
        colors = cfg.colormap(pred).numpy()

        (
            Path(output_file)
            .parent
            .mkdir(parents=True, exist_ok=True)
        )
        tmp = (
                Path(output_file)
                .parent
                / f'tmp.{Path(output_file).name}'
        )

        try:
            if not cv2.imwrite(str(tmp), colors):
                raise RuntimeError(f'cv2.imwrite failed for {output_file}')
            os.replace(tmp, output_file)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return output_file

    @classmethod
    def _compute_sidebyside(
            cls,
            static: str,
            colorized_file: str,
            output_file: str,
    ) -> str:
        """Generate side-by-side composite from input and colorized prediction files."""

        input_image = Image.open(static).convert('RGB')
        colorized_image = Image.open(colorized_file).convert('RGB')

        size = input_image.width * 2, input_image.height
        composited = Image.new('RGB', size)
        composited.paste(input_image, (0, 0))
        composited.paste(colorized_image, (input_image.width, 0))

        (
            Path(output_file)
            .parent
            .mkdir(parents=True, exist_ok=True)
        )
        tmp = (
                Path(output_file)
                .parent
                / f'tmp.{Path(output_file).name}'
        )

        try:
            composited.save(str(tmp))
            os.replace(tmp, output_file)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return output_file

    @classmethod
    def _compute_intensity(
            cls,
            prob_file: str,
            pred_file: str,
            output_file: str,
    ) -> str:
        """Generate intensity visualization from prob and pred files."""
        prob = tifffile.imread(prob_file)
        pred = tifffile.imread(pred_file)
        prob = torch.from_numpy(prob)
        pred = torch.from_numpy(pred)
        colors = cfg.colormap(pred)

        keep = torch.arange(prob.size(0))
        keep = keep[keep != cfg.dataset.ignore_label]
        fore_prob, _ = (
            torch.index_select(prob, 0, keep)
            .max(dim=0)
        )

        fore_colors = (
            colors
            .to(torch.float32)
            .mul_(fore_prob.unsqueeze(-1))
            .round_()
            .clamp_(0., 255.)
            .to(torch.uint8)
        )

        black_pixel_mask = (
            colors
            .sum(dim=-1, keepdim=True)
            .eq(0)
        )
        intensity = torch.where(black_pixel_mask, fore_colors, colors)

        (
            Path(output_file)
            .parent
            .mkdir(parents=True, exist_ok=True)
        )
        tmp = (
                Path(output_file)
                .parent
                / f'tmp.{Path(output_file).name}'
        )

        try:
            if not cv2.imwrite(str(tmp), intensity.numpy()):
                raise RuntimeError(f'cv2.imwrite failed for {output_file}')
            os.replace(tmp, output_file)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return output_file

    @classmethod
    def _compute_overlay(
            cls,
            static: str,
            colored_file: str,
            output_file: str,
            opacity: float = 0.20,
    ) -> str:
        """
        Generate overlay by blending colored prediction onto input image.

        Uses 20% opacity for colored predictions, with 0% opacity for black pixels
        from the colormaps. More efficient than the VecGrid._overlay implementation
        using PIL's blend and paste with alpha mask.

        Args:
            static: Path to input image
            colored_file: Path to colored prediction
            output_file: Path to save overlay
            opacity: Opacity for non-black pixels (default: 0.20)

        Returns:
            Path to the created overlay file
        """
        if not 0.0 <= opacity <= 1.0:
            raise ValueError('opacity must be within [0, 1]')

        input_image = Image.open(static).convert('RGB')
        colored_image = Image.open(colored_file).convert('RGB')

        # create mask for non-black pixels
        mask: np.ndarray = (
            np.array(colored_image)
            .sum(axis=-1)
            .astype(bool)
        )

        # create alpha mask: 0 for black pixels, opacity*255 for others
        mask: np.ndarray
        alpha = Image.fromarray(
            mask
            .__mul__(opacity * 255)
            .astype(np.uint8),
            mode='L'
        )

        # blend using PIL's composite with alpha mask
        overlay = Image.composite(colored_image, input_image, alpha)

        (
            Path(output_file)
            .parent
            .mkdir(parents=True, exist_ok=True)
        )
        tmp = (
                Path(output_file)
                .parent
                / f'tmp.{Path(output_file).name}'
        )

        try:
            overlay.save(str(tmp))
            os.replace(tmp, output_file)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return output_file

    @classmethod
    def _compute_soft(
            cls,
            prob_file: str,
            output_file: str,
    ) -> str:
        """Generate probability-weighted color visualization."""
        prob = tifffile.imread(prob_file)
        prob = torch.from_numpy(prob)
        num_classes = prob.size(0)

        classes = torch.arange(num_classes)
        # todo: use a custom colorscheme, with r, g, b for the classes,
        #  instead of current which will cause ambiguities
        palette = cfg.colormap(classes).float()

        # compute weighted average of colors
        # einsum: [c,h,w] x [c,n] -> [h,w,n]
        soft_image = torch.einsum('chw,cn->hwn', prob, palette)

        soft_image = (
            soft_image
            .round_()
            .clamp_(0., 255.)
            .to(torch.uint8)
            .numpy()
        )

        (
            Path(output_file)
            .parent
            .mkdir(parents=True, exist_ok=True)
        )
        tmp = (
                Path(output_file)
                .parent
                / f'tmp.{Path(output_file).name}'
        )

        try:
            # cv2 expects BGR, assuming cfg.colormap returns RGB
            if not cv2.imwrite(str(tmp), cv2.cvtColor(soft_image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f'cv2.imwrite failed for {output_file}')
            os.replace(tmp, output_file)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return output_file
