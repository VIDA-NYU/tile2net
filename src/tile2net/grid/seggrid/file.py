from __future__ import annotations

import hashlib
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image

from .postprocess import PostProcess
from .. import frame
from ..cfg import cfg
from ..grid import file
from ...grid import util

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))

if False:
    from .seggrid import SegGrid


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


class File(
    file.File
):
    grid: SegGrid

    @PostProcess
    def postprocess(self) -> pd.Series:
        """
        Namespace for work-in-progress postprocessing of segmentation results.
        """

    @frame.column
    def infile(self) -> pd.Series:
        """
        A file for each seg-tile: the stitched input grid.
        Stitches input files when seggrid.file is accessed
        """
        grid = self.grid
        files = grid.ingrid.outdir.seggrid.infile.files(grid)
        self.infile = files
        if not files.map(os.path.exists).all():
            ingrid = grid.ingrid
            _ = ingrid.file.infile
            mosaics = ingrid.segtile.infile
            assert (
                ingrid.file.infile
                .map(os.path.exists)
                .all()
            )
            ingrid._stitch_to_file(
                tiles=ingrid.file.infile,
                mosaics=mosaics,
                row=ingrid.segtile.row,
                col=ingrid.segtile.col,
            )
            msg = f"Files not stitched: {files[~files.map(os.path.exists)]}"
            assert files.map(os.path.exists).all(), msg

        return files

    @frame.column
    def pred(self) -> pd.Series:
        """
        File-paths to segmentation masks where each pixel value represents a class ID.

        Core output of the segmentation pipeline. Each pixel in the mask corresponds
        to a semantic class.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.pred
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid.broadcast
        files = grid.outdir.seggrid.pred.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.pred = files
        self.pred = files
        if (
                not grid.predict
                and not files.map(os.path.exists).all()
        ):
            grid.predict(probs=False)
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def prob(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.colorized
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid.broadcast
        files = grid.outdir.seggrid.prob.files(grid)
        loc = ~files.index.duplicated()
        files = files.loc[loc]
        grid.file.prob = files
        self.prob = files
        if (
                not bool(grid.predict)
                and not files.map(os.path.exists).all()
        ):
            grid.predict(probs=True)
            assert files.map(os.path.exists).all()
        return files

    @frame.column
    def colorized(self) -> pd.Series:
        """
        File-paths to color-coded segmentation masks for visualization.

        Lazily generated from prob and pred files already saved to disk.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.colorized
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid
        FILES = grid.ingrid.outdir.seggrid.colorized.files(grid)
        self.colorized = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            probs = self.prob[loc]
            preds = self.pred[loc]
            files = FILES[loc]
            max_workers = min(self.grid.cfg.compress_workers, len(files))
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
    def error(self) -> pd.Series:
        """
        File-paths to error masks comparing predictions to ground truth.

        Lazily generated from prob and pred files already saved to disk.

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.error
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        raise NotImplementedError('requires GT')
        grid = self.grid
        FILES = grid.ingrid.outdir.seggrid.error.files(grid)
        self.error = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            probs = self.prob[loc]
            preds = self.pred[loc]
            files = files[loc]

            max_workers = min(self.grid.cfg.compress_workers, len(files))
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
    def intensity(self) -> pd.Series:
        """
        File-paths to intensity representation of segmentation results.

        Lazily generated from prob and pred files already saved to disk.
        Alternative visualization format where segmentation classes are represented
        as intensity values.

        Returns:
            pd.Series: File paths to intensity representations for each seg-tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.intensity
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid
        FILES = grid.ingrid.outdir.seggrid.intensity.files(grid)
        self.intensity = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            files = FILES.loc[loc]
            probs = self.prob.loc[loc]
            preds = self.pred.loc[loc]

            max_workers = min(len(files), self.grid.cfg.compress_workers)
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

        Lazily generated from infile and colorized files already saved to disk.
        Creates composite images with the original input on the left and the
        colorized prediction on the right.

        Returns:
            pd.Series: File paths to side-by-side composites for each seg-tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.sidebyside
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid
        FILES = grid.ingrid.outdir.seggrid.sidebyside.files(grid)
        self.sidebyside = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            files = FILES.loc[loc]
            infiles = self.infile.loc[loc]
            colorized = self.colorized.loc[loc]

            max_workers = min(len(files), self.grid.cfg.compress_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                it = zip(infiles, colorized, files)
                futures: dict[Future, str] = {
                    threads.submit(self._compute_sidebyside, infile, colorized_file, file):
                        file
                    for infile, colorized_file, file in it
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

        Lazily generated from infile and colored files already saved to disk.
        Creates overlay images with 20% opacity colorized predictions on top of
        original input, with 0% opacity for black pixels from the colormaps.

        Returns:
            pd.Series: File paths to overlay images for each seg-tile

        Example:
            >>> ingrid: InGrid
            >>> ingrid.seggrid.file.overlay
            xtile  ytile
            79320  96960    /home/<user>/tile2net/ma/Boston Common, MA/s...
        """
        grid = self.grid
        FILES = grid.ingrid.outdir.seggrid.overlay.files(grid)
        self.overlay = FILES

        loc = ~FILES.map(os.path.exists)
        if loc.any():
            files = FILES.loc[loc]
            infiles = self.infile.loc[loc]
            colored = self.colored.loc[loc]

            max_workers = min(len(files), self.grid.cfg.compress_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                it = zip(infiles, colored, files)
                futures: dict[Future, str] = {
                    threads.submit(self._compute_overlay, infile, colored_file, file):
                        file
                    for infile, colored_file, file in it
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
    def disk_usage(self):
        result = util.path2fsize(self.pred)
        result += util.path2fsize(self.colorized)
        return result

    @staticmethod
    def _load_prob(prob_file: str) -> torch.Tensor:
        """Load probability map from TIFF file. Returns tensor with shape (C, H, W)."""
        arr = tifffile.imread(prob_file).astype(np.float32)
        return torch.from_numpy(arr)

    @staticmethod
    def _load_pred(pred_file: str) -> torch.Tensor:
        """Load prediction mask from PNG file. Returns tensor with shape (H, W)."""
        arr = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(arr)

    @classmethod
    def _compute_colorized(
            cls,
            prob_file: str,
            pred_file: str,
            output_file: str,
    ) -> str:
        """Generate colorized visualization from prob and pred files."""
        pred = cls._load_pred(pred_file)
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
            infile: str,
            colorized_file: str,
            output_file: str,
    ) -> str:
        """Generate side-by-side composite from input and colorized prediction files."""

        input_image = Image.open(infile).convert('RGB')
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
        prob = cls._load_prob(prob_file)
        pred = cls._load_pred(pred_file)
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
            infile: str,
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
            infile: Path to input image
            colored_file: Path to colored prediction
            output_file: Path to save overlay
            opacity: Opacity for non-black pixels (default: 0.20)
        
        Returns:
            Path to the created overlay file
        """
        if not 0.0 <= opacity <= 1.0:
            raise ValueError('opacity must be within [0, 1]')

        input_image = Image.open(infile).convert('RGB')
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

