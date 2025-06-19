print('⚠️AI GENERATED🤖')

from typing import TypeVar, Union, Optional

import numpy as np
import torch
from PIL import Image

if False:
    from tile2net.tiles.tiles import Tiles  # noqa: F401

T = TypeVar('T')


class ColorMap:
    def __init__(self, palette: list[int] | None = None) -> None:
        if palette is None or callable(palette):
            palette = [
                0, 0, 255,   # class 0 → blue
                0, 128, 0,   # class 1 → green
                255, 0, 0,   # class 2 → red
                0, 0, 0      # class 3 → black
            ]
        if len(palette) % 3 != 0:
            raise ValueError("Palette length must be a multiple of 3.")
        if len(palette) > 256 * 3:
            raise ValueError("Palette may not exceed 256 colours (768 values).")

        self.palette: list[int] = palette + [0] * (256 * 3 - len(palette))
        self._lut_np = np.asarray(self.palette, dtype=np.uint8).reshape(-1, 3)
        self._lut_torch = torch.tensor(self.palette, dtype=torch.uint8).view(-1, 3)

    def __call__(self, item: Union[Image.Image, np.ndarray, torch.Tensor, T]) -> T:
        # PIL.Image ----------------------------------------------------------
        if isinstance(item, Image.Image):
            out = item.convert("P")
            out.putpalette(self.palette)
            return out  # type: ignore[return-value]

        # NumPy ndarray ------------------------------------------------------
        if isinstance(item, np.ndarray):
            if item.ndim not in (2, 3):
                raise ValueError("Expected (H,W) or (N,H,W) class-index array.")
            item_int = item.astype(np.intp)
            if item.ndim == 2:           # (H,W) → (H,W,3)
                return self._lut_np[item_int].copy()            # type: ignore[return-value]
            else:                        # (N,H,W) → (N,H,W,3)
                n, h, w = item.shape
                return self._lut_np[item_int.reshape(-1)].reshape(n, h, w, 3).copy()  # type: ignore[return-value]

        # torch.Tensor -------------------------------------------------------
        if isinstance(item, torch.Tensor):
            if item.dim() not in (2, 3):
                raise ValueError("Expected (H,W) or (N,H,W) class-index tensor.")
            if not item.dtype.is_floating_point and item.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
            ):
                raise TypeError("Tensor must contain integer class indices.")

            item_long = item.to(torch.long)
            if item.dim() == 2:          # (H,W) → (H,W,3)
                return self._lut_torch[item_long].contiguous()  # type: ignore[return-value]
            else:                        # (N,H,W) → (N,H,W,3)
                n, h, w = item.shape
                return self._lut_torch[item_long.view(-1)].view(n, h, w, 3).contiguous()  # type: ignore[return-value]

        raise TypeError(f"Unsupported type: {type(item).__name__}")

    def __set_name__(self, owner, name):
        self.__name__ = name
