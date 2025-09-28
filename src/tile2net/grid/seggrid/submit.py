from __future__ import annotations

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
        finally:
            try:
                self.threads.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
            # free lists
            self.prev.clear()
            self.next.clear()

    def _imwrite(
            self,
            filename: str,
            img,
            params
    ):
        p = Path(filename)
        parent = p.parent
        parent.mkdir(exist_ok=True)
        tmp = p.parent / f'tmp.{p.name}'
        # fd, tmp = tempfile.mkstemp(
        #     prefix=".tmp_",
        #     dir=str(p.parent),
        # )
        # os.close(fd)
        try:
            ok = cv2.imwrite(tmp, img, params)
            if not ok:
                raise RuntimeError(
                    f'imwrite failed for {filename} with '
                    f'shape={img.shape} dtype={img.dtype}'
                )
            os.replace(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            finally:
                raise

    def imwrite(
            self,
            filename: str,
            img,
            params=(cv2.IMWRITE_PNG_COMPRESSION, 1),
    ):
        """
        Schedule saving an image to disk with atomic replace.
        """
        future = self.threads.submit(self._imwrite, filename, img, params)
        self.next.append(future)

    def _to_npy(
            self,
            filename: str,
            arr: np.ndarray,
    ) -> None:
        p = Path(filename)
        parent = p.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f'tmp.{p.name}'

        try:
            # Save array to a temporary .npy file
            np.save(tmp, arr)
            os.replace(tmp, p)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            finally:
                raise

    def to_npy(
            self,
            filename: str,
            arr: np.ndarray,
    ):
        """
        Schedule saving a numpy array to .npy with atomic replace.
        """
        future = self.threads.submit(self._to_npy, filename, arr)
        self.next.append(future)
