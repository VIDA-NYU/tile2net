from __future__ import annotations

from typing import *

import numpy as np
import torch

if False:
    from .stitch import StitchDataSet

T = TypeVar("T", np.ndarray, torch.Tensor)


class BaseDataLoader(
    torch.utils.data.DataLoader,
    Generic[T]
):
    dataset: StitchDataSet

    @property
    def wrapper(self):
        return self.dataset.wrapper

    if False:
        def __iter__(self) -> Iterator[T]:
            return super().__iter__()

    def __repr__(self):
        result = self.__class__.__name__
        result += '\n\n'
        result += self.wrapper.__repr__()
        return result


class DataLoader(
    BaseDataLoader[np.ndarray],
):
    """Explicitly typed DataLoader yielding numpy.ndarray"""


class TensorDataLoader(
    BaseDataLoader[torch.Tensor],
):
    """Explicitly typed DataLoader yielding torch.Tensor"""
