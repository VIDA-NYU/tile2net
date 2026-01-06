from __future__ import annotations
import tifffile

import os
import time
from concurrent.futures import wait, ThreadPoolExecutor
from functools import *
from pathlib import Path

import cv2
import numpy as np


class Submit:
    def __init__(
            self,
            workers: int = 4,
    ):
        """
        Initialize a Submit scheduler with a thread pool.
        """
        self.workers = workers

    @cached_property
    def threads(self):
        """
        Thread pool used to execute background tasks.
        """
        return ThreadPoolExecutor(max_workers=self.workers)

    @cached_property
    def prev(self):
        return []

    @cached_property
    def next(self):
        return []

    @cached_property
    def period(self):
        return 1800

    @cached_property
    def errors(self):
        return []

    @cached_property
    def t(self) -> float:
        return time.time()
        # context manager to guarantee shutdown

    def __enter__(self) -> "Submit":
        return self

    def __exit__(
            self,
            exc_type,
            exc,
            tb,
    ) -> None:
        try:
            # drain both prev and next
            futs = []
            futs.extend(self.prev)
            futs.extend(self.next)
            if futs:
                done, _ = wait(futs)
                for f in done:
                    try:
                        _ = f.result()
                    except BaseException as e:
                        self.errors.append(e)

            if self.errors:
                # raise the first error that crashed a subprocess
                raise self.errors[0]

        finally:
            try:
                self.threads.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass

            # free the threads
            self.prev.clear()
            self.next.clear()


    def rotate(self) -> None:
        """
        Wait for previous batch to complete, then rotate next -> prev.
        Call this between minibatch iterations to ensure we don't get
        too far ahead of disk I/O.
        """
        # wait for prev batch to finish
        for f in self.prev:
            try:
                f.result()
            except BaseException as e:
                self.errors.append(e)
        self.prev.clear()
        
        # move next -> prev
        self.prev.extend(self.next)
        self.next.clear()

    def to_tiff(
            self,
            filename: str,
            arr: np.ndarray,
    ) -> None:
        p = Path(filename)
        parent = p.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f'tmp.{p.name}'

        try:
            tifffile.imwrite(tmp, arr, compression='zlib')
            os.replace(tmp, p)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            finally:
                raise

    def imwrite(
            self,
            filename: str,
            img,
            params: list
    ):
        p = Path(filename)
        parent = p.parent
        parent.mkdir(exist_ok=True)
        tmp = (
            p.parent
            .joinpath(f'tmp.{p.name}')
            .__str__()
        )
        if not params:
            params = None
        try:
            if not cv2.imwrite(tmp, img, params):
                raise RuntimeError(
                    f'imwrite failed for {filename} with '
                    f'shape={img.shape} dtype={img.dtype}'
                )
            os.replace(tmp, p)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            finally:
                raise