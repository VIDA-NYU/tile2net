from __future__ import annotations

import concurrent.futures as cf
import os
from functools import cached_property
from pathlib import Path
from typing import *
from typing import Union

import imageio.v3 as iio
import numpy as np
import tifffile
import torch.utils.data

from tile2net.core.cfg import cfg
from tile2net.core.loaders.dataloader import BaseDataLoader
from tile2net.core.loaders.datawrapper import DataWrapper

if TYPE_CHECKING:
    from tile2net.core.grid.grid import Grid


class Propagate:
    def __set_name__(self, owner: StitchDataSet, name):
        owner.to_propagate.add(name)
        setattr(owner, name, self.wrapped)
        if hasattr(self.wrapped, '__set_name__'):
            self.wrapped.__set_name__(owner, name)

    def __init__(self, wrapped):
        self.wrapped = wrapped


class ToPropagate:
    def __set_name__(self, owner, name):
        self.__name__ = name
        self.cache = {}

    def __get__(
            self,
            instance,
            owner: type[StitchDataSet]
    ) -> set[str]:
        if owner not in self.cache:
            out = {
                name
                for base in owner.__bases__
                if hasattr(base, self.__name__)
                for name in base.to_propagate
            }
            self.cache[owner] = out
        return self.cache[owner]

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        ...


class StitchDataSet(
    torch.utils.data.Dataset,
):
    """
    Implements __getitem__ and __len__ as required by torch.utils.data.Dataset.
    The columns from `DataWrapper` are reshaped into lists or lists of lists,
    so that __getitem__[int] can be performed efficiently.
    """
    wrapper: DataWrapper
    """Underlying DataWrapper instance."""

    def __init__(
            self,
            wrapper: DataWrapper,
            tile_dimension: int,
            mode: str = None,
            padding: int = 0,
            channels: int = 3,
            *args,
            **kwargs,
    ):
        self.mode = mode
        self.channels = channels
        self.wrapper = wrapper
        self.tile_dimension = tile_dimension
        self.padding = padding

    @ToPropagate
    def to_propagate(self):
        ...

    @cached_property
    def threads(self):
        """Number of threads to use for reading input tiles."""
        return 1

    @cached_property
    def read(self) -> Callable[[str], np.ndarray]:
        it = (
            Path(file)
            .suffix
            .lower()
            for file in self.wrapper.image_paths.values
            if file is not None
        )
        ext = next(it, None)
        match ext:
            case '.npy':
                return self.read_npy
            case '.tif' | '.tiff':
                return self.read_tiff
            case _:
                return self.read_iio

    def __len__(self):
        """number of mosaics"""
        result = len(self.index)
        return result

    @cached_property
    def index(self) -> list[Any]:
        """mosaic identifiers; could be integer or destination path"""
        result = (
            self.wrapper
            .index
            .unique()
        )
        return result

    @cached_property
    def row(self) -> list[list[int]]:
        """Row in the mosaic which the tile comprises."""
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names, sort=False)
            .row
            .apply(list)
            .tolist()
        )
        return result

    @cached_property
    def col(self) -> list[list[int]]:
        """Column in the mosaic which the tile comprises."""
        result = (
            self.wrapper.frame
            .groupby(level=self.wrapper.frame.index.names, sort=False)
            .col
            .apply(list)
            .tolist()
        )
        return result

    @cached_property
    def nrow(self) -> int:
        """Number of rows for the mosaic which the tile comprises."""
        nrow = (
                self.wrapper
                .frame
                .groupby(level=self.wrapper.frame.index.names, sort=False)
                .row
                .max()
                + 1
        )
        if nrow.nunique() > 1:
            raise ValueError('mosaic rows must be identical')
        return nrow.iloc[0]

    @cached_property
    def ncol(self) -> int:
        """Number of columns for the mosaic which the tile comprises."""
        ncol = (
                self.wrapper
                .frame
                .groupby(level=self.wrapper.frame.index.names, sort=False)
                .col
                .max()
                + 1
        )
        if ncol.nunique() > 1:
            raise ValueError('mosaic columns must be identical')
        return ncol.iloc[0]

    @cached_property
    def image_paths(self) -> list[list[str]]:
        """Input static imagery file path for each tile in each mosaic."""
        if 'image_paths' not in self.wrapper.frame.columns:
            raise ValueError('image_path column is required in DataWrapper')
        result = (
            self.wrapper
            .frame
            .groupby(level=self.wrapper.frame.index.names, sort=False)
            .image_paths
            .apply(list)
            .tolist()
        )
        return result

    @Propagate
    @cached_property
    def colorized_paths(self) -> list[list[str]] | None:
        """Colorized output file path for each tile in each mosaic."""
        if 'colorized_paths' not in self.wrapper.frame.columns:
            return None
        result = (
            self.wrapper
            .frame
            .groupby(level=self.wrapper.frame.index.names, sort=False)
            .colorized_paths
            .apply(list)
            .tolist()
        )
        return result

    @Propagate
    @cached_property
    def pred_paths(self) -> list[str]:
        """Prediction output file path for each mosaic."""
        if 'pred_paths' not in self.wrapper.frame.columns:
            return None
        result = (
            self.wrapper
            .frame
            .groupby(level=self.wrapper.frame.index.names, sort=False)
            .pred_paths
            .first()
            .tolist()
        )
        return result

    @Propagate
    @cached_property
    def prob_paths(self) -> list[str]:
        """Probability output file path for each mosaic."""
        if 'prob_paths' not in self.wrapper.frame.columns:
            return None
        result = (
            self.wrapper
            .frame
            .groupby(level=self.wrapper.frame.index.names, sort=False)
            .prob_paths
            .first()
            .tolist()
        )
        return result

    @Propagate
    @cached_property
    def unclipped_prob_paths(self) -> list[str] | None:
        """Unclipped probability output file path for each mosaic."""
        if 'unclipped_prob_paths' not in self.wrapper.frame.columns:
            return None
        result = (
            self.wrapper
            .frame
            .groupby(level=self.wrapper.frame.index.names, sort=False)
            .unclipped_prob_paths
            .first()
            .tolist()
        )
        return result

    @cached_property
    def crop_size(self) -> Union[int, Tuple[int, int]]:
        result = cfg.dataset.crop_size
        if isinstance(result, (list, tuple)):
            ...
        else:
            result = int(result)
        return result

    @cached_property
    def mean_std(self) -> tuple[
        list[float],
        list[float]
    ]:
        mean = cfg.dataset.mean
        std = cfg.dataset.std
        result = mean, std
        return result

    @cached_property
    def sample(self) -> np.ndarray:
        """
        A sample of the input imagery, to automatically determine the dimension of the imagery.
        This assumes all input tiles have the same shape.
        """
        try:
            path: str = next(
                path
                for paths in self.image_paths
                for path in paths
                if Path(path).is_file()
            )
        except StopIteration:
            raise FileNotFoundError('No existing input tiles to infer shape from.')
        except Exception:
            raise
        sample = self.read(path)
        if sample.ndim != self.channels:
            raise ValueError(f'{path!s}: unexpected ndim {sample.ndim}')
        return sample

    @staticmethod
    def read_npy(path: str) -> np.ndarray:
        arr = np.load(path, mmap_mode=None)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f'{path!s}: expected HxWxC array, got {arr.shape}')
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if arr.shape[2] == 3:
            pad = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
            arr = np.concatenate((arr, pad), axis=2)
        return arr

    @staticmethod
    def read_iio(path: str) -> np.ndarray:
        arr = iio.imread(path)

        match arr.ndim:
            case 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            case 3:
                pass
            case _:
                raise ValueError(f'{path!s}: unsupported image ndim {arr.ndim}')

        match arr.dtype:
            case np.float32 | np.float64:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            case np.uint8:
                pass
            case _:
                arr = arr.astype(np.uint8, copy=False)

        match arr.shape[2]:
            case 1:
                arr = np.repeat(arr, 3, axis=2)
            case 3:
                pad = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
                arr = np.concatenate((arr, pad), axis=2)
            case 4:
                pass
            case _:
                raise ValueError(f'{path!s}: expected channels in (1,3,4), got {arr.shape[2]}')

        return arr

    @staticmethod
    def read_tiff(path: str) -> np.ndarray:
        arr = tifffile.imread(path)

        match arr.ndim:
            case 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            case 3:
                pass
            case _:
                raise ValueError(f'{path!s}: unsupported image ndim {arr.ndim}')

        match arr.dtype:
            case np.float32 | np.float64:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            case np.uint8:
                pass
            case _:
                arr = arr.astype(np.uint8, copy=False)

        match arr.shape[2]:
            case 1:
                arr = np.repeat(arr, 3, axis=2)
            case 3:
                pad = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
                arr = np.concatenate((arr, pad), axis=2)
            case 4:
                pass
            case _:
                raise ValueError(f'{path!s}: expected channels in (1,3,4), got {arr.shape[2]}')

        return arr

    @cached_property
    def h(self):
        """Height of the mosaic in pixels. This assumes all input tiles have the same shape."""
        return self.nrow * self.sample.shape[0]

    @cached_property
    def w(self):
        """Width of the mosaic in pixels. This assumes all input tiles have the same shape."""
        return self.ncol * self.sample.shape[1]

    @cached_property
    def _mosaic_pool(self):
        """Pre-allocated mosaic buffer pool for memory reuse.
        Improves performance by avoiding malloc and free under-the-hood.
        """
        return np.empty((self.h, self.w, self.channels), dtype=np.uint8)

    @property
    def mosaic(self):
        """Get a zeroed mosaic array from the pool.
        Reuses the same underlying buffer to avoid repeated allocations.
        """
        out = self._mosaic_pool
        out.fill(0)
        return out

    def __getitem__(self, item, ) -> np.ndarray:
        # composite tiles into a single RGB mosaic

        # pull grouped lists
        files = self.image_paths[item]
        rows = self.row[item]
        cols = self.col[item]
        index = self.index[item]
        mosaic = self.mosaic

        if self.threads == 1:
            for f, r, c in zip(files, rows, cols, strict=True):
                if f is None:
                    # skip empty tiles
                    continue
                arr = self.read(f)
                self.paste(mosaic, arr, r, c)

        else:
            # submit read tasks for each tile except None
            tasks: list[tuple[int, str]] = [
                (i, f)
                for i, f in enumerate(files)
                if f is not None
            ]

            results: dict[int, np.ndarray] = {}
            max_workers = self.threads

            with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(self.read, f): i
                    for i, f in tasks
                }
                for fut in cf.as_completed(futs):
                    idx = futs[fut]
                    results[idx] = fut.result()

            # sequentially paste following the original order
            for i, (f, r, c) in enumerate(zip(files, rows, cols, strict=True)):
                if f is None:
                    continue
                arr = results[i]
                self.paste(mosaic, arr, r, c)

        out = mosaic

        crop = (
                self.tile_dimension
                - self.padding
                % self.tile_dimension
        )
        out = out[
            crop:out.shape[0] - crop,
            crop:out.shape[1] - crop,
        ]

        return out

    def paste(
            self,
            mosaic: np.ndarray,
            tile: np.ndarray,
            row: int,
            col: int,
    ) -> None:
        """
        Paste a tile into the mosaic at the given row/column position.
        Uses alpha channel compositing: only pixels where alpha != 0 are copied.

        Based on the original _paste implementation from __getitem__.
        """
        tile_h = tile.shape[0]
        tile_w = tile.shape[1]

        # compute region of interest
        y0 = row * tile_h
        x0 = col * tile_w
        sl = np.s_[y0:y0 + tile_h, x0:x0 + tile_w, :]

        # split channels and composite where alpha != 0
        rgb = tile[..., :3]
        a = tile[..., 3:4]
        dst = mosaic[sl]
        np.copyto(dst, rgb, where=(a != 0))

    def loader[T: BaseDataLoader](
            self,
            batch_size=None,
            shuffle=None,
            drop_last=None,
            sampler=None,
            num_workers=None,
            pin_memory=None,
            persistent_workers=None,
            worker_init_fn=None,
            cls=BaseDataLoader,
            **kwargs
    ) -> Union[T, BaseDataLoader]:
        """
        Convenient constructor of a DataLoader for this dataset.

        See also:
            >>> Grid._stitch2file
        """
        if batch_size is None:
            batch_size = os.cpu_count()
        if shuffle is None:
            shuffle = False
        if drop_last is None:
            drop_last = False
        if sampler is None:
            sampler = None
        if num_workers is None:
            num_workers = 0
        if pin_memory is None:
            pin_memory = True
        if persistent_workers is None:
            persistent_workers = False

        out = cls(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init_fn,
        )
        return out


class TensorDataSet(
    StitchDataSet
):
    """
    DataSet extension which returns torch.Tensor from __getitem__
    """

    def __getitem__(
            self,
            item,
    ) -> torch.Tensor:
        arr = super().__getitem__(item)
        result = (
            torch
            .from_numpy(arr)
            .permute(2, 0, 1)
            .contiguous()
        )
        return result


class StitchWriterDataSet(
    StitchDataSet
):
    def __init__(
            self,
            wrapper: DataWrapper,
            write: Callable[[np.ndarray, str], str] | None = None,
            *args,
            **kwargs,
    ):
        super().__init__(wrapper, *args, **kwargs)
        if write:
            self.write = write

    def __getitem__(self, item):
        out = super().__getitem__(item)
        index = self.index[item]
        self.write(out, index)
        return -1

    @cached_property
    def write(self) -> Callable[[np.ndarray, str], str]:
        it = (
            Path(file)
            .suffix
            .lower()
            for file in self.wrapper.index.values
            if file is not None
        )
        ext = next(it, None)
        match ext:
            case '.tif' | '.tiff':
                return self.write_tiff
            case _:
                return self.write_iio

    @staticmethod
    def write_iio(
            arr: np.ndarray,
            file: str,
    ):
        outp = Path(file)
        outp.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(outp, arr)
        return str(outp)

    @staticmethod
    def write_tiff(
            arr: np.ndarray,
            file: str,
    ):
        outp = Path(file)
        outp.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(outp, arr)
        return str(outp)
