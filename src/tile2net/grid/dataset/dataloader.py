import torch.utils.data

from types import *
from typing import *
import numpy as np
import torch

from types import *
from typing import *
import numpy as np

T = TypeVar("T", np.ndarray, torch.Tensor)


class BaseDataLoader(
    torch.utils.data.DataLoader,
    Generic[T]
):
    if False:
        def __iter__(self) -> Iterator[T]:
            return super().__iter__()


class DataLoader(
    BaseDataLoader[np.ndarray],
):
    """Explicitly typed DataLoader yielding numpy.ndarray"""


class TensorDataLoader(
    BaseDataLoader[torch.Tensor],
):
    """Explicitly typed DataLoader yielding torch.Tensor"""
