from __future__ import annotations

import concurrent.futures as cf
from functools import cached_property
from typing import *
from typing import NamedTuple

import imageio.v3 as iio
import numpy as np
import requests
from requests.adapters import HTTPAdapter

from tile2net.core.loaders.stitch import StitchDataSet, DataWrapper

if TYPE_CHECKING:
    pass


class FetchTileResult(NamedTuple):
    status: Literal['success', 'empty', 'not_found', 'failed'] | str
    array: np.ndarray | None


class StreamStitchDataSet(
    StitchDataSet
):
    """
    Stitches mosaics from remote URLs or local paths concurrently.
    Returns a dict containing the mosaic and download statistics.
    """

    def __init__(
            self,
            wrapper: DataWrapper,
            tile_dimension: int,
            mode: str,
            padded_dimension: int = None,
            *args,
            **kwargs,
    ):
        """
        :param wrapper: The DataWrapper containing grid information.
        :param tile_shape: The (H, W, C) shape of individual tiles.
                           Required to initialize buffers if all sources are empty (204).
        """
        self.mode = mode
        super().__init__(
            wrapper,
            tile_dimension=tile_dimension,
            mode=mode,
            padded_dimension=padded_dimension,
            *args,
            **kwargs
        )
        self.tile_dimension = tile_dimension
        self.padded_dimension = padded_dimension
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    @cached_property
    def sample(self) -> np.ndarray:
        """
        Override to return dummy array based on init shape.
        Prevents base class from hitting disk to infer dimensions.
        """
        shape = (self.tile_dimension, self.tile_dimension, self.channels)
        return np.zeros(shape, dtype=np.uint8)

    @cached_property
    def threads(self):
        return 16

    def _fetch_tile(
            self,
            source: str,
    ) -> FetchTileResult:
        if not source.startswith(('http://', 'https://')):
            # Local file fallback
            try:
                arr = self.read(source)
            except Exception:
                return FetchTileResult('failed', None)
            else:
                return FetchTileResult('success', arr)

        # Remote file
        try:
            with self.session.get(source, timeout=(3, 10)) as resp:
                status = resp.status_code

                if status == 204:
                    return FetchTileResult('empty', None)
                if status in (403, 404):
                    return FetchTileResult('not_found', None)
                if status != 200:
                    return FetchTileResult('failed', None)

                content = resp.content
                try:
                    # Try reading without extension first
                    arr = iio.imread(content, index=None)
                except (ValueError, IndexError):
                    # Fallback to common formats
                    try:
                        arr = iio.imread(content, extension='.png', index=None)
                    except Exception:
                        arr = iio.imread(content, extension='.jpg', index=None)

                # Normalize dimensions for pasting
                if arr.ndim == 2:
                    arr = np.repeat(arr[..., None], 3, axis=2)

                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                elif arr.shape[2] == 3:
                    # Add alpha channel for pasting logic
                    pad = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
                    arr = np.concatenate((arr, pad), axis=2)

                return FetchTileResult('success', arr.astype(np.uint8, copy=False))

        except Exception:
            return FetchTileResult('failed', None)

    def __getitem__(self, item) -> dict[str, Any]:
        """
        Returns:
            dict: {
                'mosaic': np.ndarray (H, W, 3),
                'success': int,
                'empty': int,
                'not_found': int,
                'failed': int
            }
        """
        files = self.input_paths[item]
        rows = self.row[item]
        cols = self.col[item]

        # Get a fresh buffer from the pool
        mosaic = self.mosaic

        tasks = [
            (f, r, c)
            for f, r, c in zip(files, rows, cols)
            if f is not None
        ]

        stats = {
            'success': 0,
            'empty': 0,
            'not_found': 0,
            'failed': 0,
        }

        with cf.ThreadPoolExecutor(max_workers=self.threads) as ex:
            futures = {
                ex.submit(self._fetch_tile, f): (r, c)
                for f, r, c in tasks
            }

            for fut in cf.as_completed(futures):
                r, c = futures[fut]
                try:
                    status, arr = fut.result()
                    stats[status] += 1

                    if (
                        status == 'success'
                        and arr is not None
                    ):
                        self.paste(mosaic, arr, r, c)
                except Exception:
                    stats['failed'] += 1

        a = (
            self.tile_dimension
            - self.padding
            % self.tile_dimension
        )
        mosaic = mosaic[
            a:mosaic.shape[0] - a,
            a:mosaic.shape[1] - a,
        ]

        out = dict(
            mosaic=mosaic,
            success=stats['success'],
            empty=stats['empty'],
            not_found=stats['not_found'],
            failed=stats['failed'],
        )
        return out
