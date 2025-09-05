from typing import TypeVar

import torch.utils.data

T = TypeVar("T", bound="DataSet")




class DataLoader(
    torch.utils.data.DataLoader
):
    ...
